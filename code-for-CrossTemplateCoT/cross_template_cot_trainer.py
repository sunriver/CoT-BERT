import sys
import os
import math
import time
import collections
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

from packaging import version

from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import (
    HPSearchBackend,
    PREFIX_CHECKPOINT_DIR,
    TrainOutput,
    speed_metrics,
    set_seed,
)
from transformers.utils import WEIGHTS_NAME, is_apex_available, is_torch_tpu_available, logging

# SentEval paths (同TwoStageCoT保持一致)
PATH_TO_SENTEVAL = "../SentEval"
PATH_TO_DATA = "../SentEval/data"

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval  # noqa: E402


logger = logging.get_logger(__name__)

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm  # noqa: F401
    import torch_xla.debug.metrics as met  # noqa: F401

if is_apex_available():
    from apex import amp  # noqa: F401

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast  # noqa: F401


class CrossTemplateCoTTrainer(Trainer):
    """
    Cross-Template CoT 训练器：继承Trainer，实现SentEval评估与自定义checkpoint保存
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sharded_dpp = False
        self.use_amp = False

    # ------------------------
    # 损失计算（训练阶段）
    # ------------------------
    def compute_loss(self, model, inputs, return_outputs: bool = False):
        """
        计算Cross-Template CoT损失
        """
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self.tokenizer

        anchor_template = getattr(
            self.model_args,
            "mask_embedding_sentence_template",
            'The sentence of "[X]" means [MASK], so it can be summarized as [MASK].',
        )
        positive_template = getattr(
            self.model_args,
            "mask_embedding_sentence_different_template",
            'The sentence ："[X]" means [MASK], so it can be summarized as [MASK].',
        )
        negative_template = getattr(
            self.model_args,
            "mask_embedding_sentence_negative_template",
            'The sentence ："[X]" does not mean [MASK], so it cannot be summarized as [MASK]',
        )

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids", None),
            anchor_template=anchor_template,
            positive_template=positive_template,
            negative_template=negative_template,
            tokenizer=tokenizer,
        )

        loss = outputs.loss if hasattr(outputs, "loss") else None
        return (loss, outputs) if return_outputs else loss

    # ------------------------
    # SentEval 评估
    # ------------------------
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:
        """
        使用 SentEval 进行STS/Transfer评估，返回主要指标用于早停与选择最优模型。
        """

        def prepare(params, samples):
            return

        def batcher(params, batch):
            """
            SentEval batcher函数
            batch: 单个任务的一批句子，每个句子是token列表
            SentEval对STS任务会分别传两批句子（batch1和batch2）
            """
            # 兼容bytes编码
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode("utf-8") for word in s] for s in batch]

            sentences = [" ".join(s) for s in batch]

            use_template = (
                self.model_args
                and hasattr(self.model_args, "mask_embedding_sentence")
                and self.model_args.mask_embedding_sentence
            )

            if use_template:
                anchor_template = getattr(
                    self.model_args,
                    "mask_embedding_sentence_template",
                    'The sentence of "[X]" means [MASK], so it can be summarized as [MASK].',
                )

                templated_sentences = []
                for sent in sentences:
                    parts = anchor_template.split("[X]")
                    prefix = parts[0]
                    suffix = parts[1] if len(parts) > 1 else ""
                    templated_sentences.append(prefix + sent + suffix)

                encoded = self.tokenizer.batch_encode_plus(
                    templated_sentences,
                    return_tensors="pt",
                    padding=True,
                )

                batch_size = len(sentences)
                seq_len = encoded["input_ids"].size(1)

                batch_input = {}
                for k, v in encoded.items():
                    v = v.to(self.args.device)
                    # 形状: (batch_size, 1, seq_len)
                    batch_input[k] = v.view(batch_size, 1, seq_len)
            else:
                encoded = self.tokenizer.batch_encode_plus(
                    sentences,
                    return_tensors="pt",
                    padding=True,
                )
                batch_input = {}
                for k, v in encoded.items():
                    v = v.to(self.args.device)
                    batch_input[k] = v.unsqueeze(1)

            anchor_template = getattr(
                self.model_args,
                "mask_embedding_sentence_template",
                'The sentence of "[X]" means [MASK], so it can be summarized as [MASK].',
            )

            with torch.no_grad():
                outputs = self.model(
                    **batch_input,
                    output_hidden_states=True,
                    return_dict=True,
                    sent_emb=True,
                    anchor_template=anchor_template,
                    tokenizer=self.tokenizer,
                )
                pooler_output = outputs.pooler_output

            return pooler_output.cpu()

        # SentEval配置
        params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 5}
        params["classifier"] = {"nhid": 0, "optim": "rmsprop", "batch_size": 128, "tenacity": 3, "epoch_size": 2}

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ["STSBenchmark", "SICKRelatedness"]

        if eval_senteval_transfer or self.args.eval_transfer:
            tasks = ["STSBenchmark", "SICKRelatedness", "MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]

        self.model.eval()
        results = se.eval(tasks)

        stsb_spearman = results["STSBenchmark"]["dev"]["spearman"][0]
        sickr_spearman = results["SICKRelatedness"]["dev"]["spearman"][0]

        metrics = {
            "eval_stsb_spearman": stsb_spearman,
            "eval_sickr_spearman": sickr_spearman,
            "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2,
        }

        if eval_senteval_transfer or self.args.eval_transfer:
            avg_transfer = 0.0
            for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
                avg_transfer += results[task]["devacc"]
                metrics[f"eval_{task}"] = results[task]["devacc"]
            avg_transfer /= 7
            metrics["eval_avg_transfer"] = avg_transfer

        self.log(metrics)
        return metrics

    # ------------------------
    # Checkpoint 保存策略（与TwoStageCoT保持一致）
    # ------------------------
    def _save_checkpoint(self, model, trial, metrics: Optional[Dict[str, float]] = None):
        """
        保存检查点：如果设置了 metric_for_best_model，则只在指标最优时保存；否则按step保存。
        """
        from transformers.trainer_pt_utils import get_model_param_count

        # 确保unwrap_model
        assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # 根据指标决定是否保存最佳模型
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"

            if metric_to_check not in metrics:
                logger.warning(
                    f"Metric {metric_to_check} not found in metrics dict. Available keys are: {list(metrics.keys())}"
                )
                metric_value = None
            else:
                metric_value = metrics[metric_to_check]

            if metric_value is not None:
                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    output_dir = self.args.output_dir
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

                    logger.info(f"Saving new best model to {output_dir} (metric: {metric_to_check} = {metric_value})")
                    self.save_model(output_dir)

                    if self.deepspeed:
                        self.deepspeed.save_checkpoint(output_dir)

                    # 保存优化器和scheduler
                    if self.sharded_dpp:
                        self.optimizer.consolidate_state_dict()

                    if is_torch_tpu_available():
                        xm.rendezvous("saving_optimizer_states")
                        xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        with warnings.catch_warnings(record=True) as caught_warnings:  # type: ignore[name-defined]
                            xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            reissue_pt_warnings(caught_warnings)
                    elif self.is_world_process_zero() and not self.deepspeed:
                        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        with warnings.catch_warnings(record=True) as caught_warnings:  # type: ignore[name-defined]
                            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            reissue_pt_warnings(caught_warnings)

                    # 保存Trainer状态
                    if self.is_world_process_zero():
                        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        else:
            # 默认按step保存checkpoint
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is not None and trial is not None:
                if self.hp_search_backend == HPSearchBackend.OPTUNA:
                    run_id = trial.number
                else:
                    from ray import tune  # type: ignore

                    run_id = tune.get_trial_id()
                run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
                output_dir = os.path.join(self.args.output_dir, run_name, checkpoint_folder)
            else:
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
                self.store_flos()

            logger.info(f"Saving model checkpoint to {output_dir}")
            self.save_model(output_dir)

            if self.deepspeed:
                self.deepspeed.save_checkpoint(output_dir)

            if self.sharded_dpp:
                self.optimizer.consolidate_state_dict()

            if is_torch_tpu_available():
                xm.rendezvous("saving_optimizer_states")
                xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                with warnings.catch_warnings(record=True) as caught_warnings:  # type: ignore[name-defined]
                    xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    reissue_pt_warnings(caught_warnings)
            elif self.is_world_process_zero() and not self.deepspeed:
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                with warnings.catch_warnings(record=True) as caught_warnings:  # type: ignore[name-defined]
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    reissue_pt_warnings(caught_warnings)

            if self.is_world_process_zero():
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            # 轮转旧的checkpoint
            if self.is_world_process_zero():
                self._rotate_checkpoints(use_mtime=True)

    # ------------------------
    # 训练主流程（复制TwoStageCoTTrainer.train以保持行为一致）
    # ------------------------
    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):  # type: ignore[name-defined]
        """
        训练主入口点（与TwoStageCoTTrainer基本一致）
        """
        from transformers.trainer_utils import get_last_checkpoint

        # 超参搜索
        self._hp_search_setup(trial)

        # 模型重新初始化
        if self.model_init is not None:
            set_seed(self.args.seed)
            model = self.call_model_init(trial)

            if not self.is_model_parallel:
                model = model.to(self.args.device)

            self.model = model
            self.model_wrapped = model
            self.optimizer, self.lr_scheduler = None, None

        # dataloader大小
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
        train_dataloader = self.get_train_dataloader()

        # 训练步数与epoch
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                num_train_epochs = math.ceil(self.args.num_train_epochs)
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
        else:
            num_train_epochs = 1
            max_steps = self.args.max_steps
            num_update_steps_per_epoch = max_steps

        # 创建优化器和scheduler
        if self.args.deepspeed:
            from transformers.integrations import deepspeed_init

            model, optimizer, lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)
            self.model = model.module
            self.model_wrapped = model
            self.deepspeed = model
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        else:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # 加载optimizer / scheduler状态
        self._load_optimizer_and_scheduler(model_path)

        model = self.model_wrapped

        if self.use_apex:  # type: ignore[attr-defined]
            model, self.optimizer = amp.initialize(  # type: ignore[name-defined]
                model, self.optimizer, opt_level=self.args.fp16_opt_level
            )

        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.sharded_dpp:
            from torch.distributed.algorithms.ddp_sharded import ShardedDDP  # type: ignore

            model = ShardedDDP(model, self.optimizer)
        elif self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )

        if model is not self.model:
            self.model_wrapped = model

        # batch大小
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )

        num_examples = self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * max_steps

        logger.info("***** Running CrossTemplateCoT Training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # 恢复checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first "
                    f"{steps_trained_in_current_epoch} batches in the first epoch."
                )

        # callback handler
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = self.hp_params(trial) if trial is not None else None  # type: ignore[attr-defined]

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(self.args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # 跳过已训练的epoch以对齐sampler随机状态
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(train_dataloader) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            assert train_dataset_is_sized, "currently we only support sized dataloader!"

            for step, inputs in enumerate(epoch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if (step + 1) % self.args.gradient_accumulation_steps != 0 and self.args.local_rank != -1:
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)

                self._total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    steps_in_epoch <= self.args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                ):
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)  # type: ignore[attr-defined]

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),  # type: ignore[attr-defined]
                                self.args.max_grad_norm,
                            )

                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        self.scaler.step(self.optimizer)  # type: ignore[attr-defined]
                        self.scaler.update()  # type: ignore[attr-defined]
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval=None)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval=None)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU configured. "
                        "Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        logger.info(
            "\n\nCrossTemplateCoT Training completed. "
            "Do not forget to share your model on huggingface.co/models =)\n\n"
        )

        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint, model_args=self.model_args)
                if not self.is_model_parallel:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint,
                    load_optimizer_states=False,
                    load_lr_scheduler_states=False,
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        self._total_loss_scalar += tr_loss.item()

        return TrainOutput(self.state.global_step, self._total_loss_scalar / max(1, self.state.global_step), metrics)


