#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将本地评估/实验产物推送到指定 Git 远程。

配置来源（择一逻辑）：

1. 显式 ``git_config_path``：若文件存在则读入（支持 JSON；路径以 .yaml/.yml 结尾时需 PyYAML）。
2. ``project_root_for_git_config``：**仅**读取 ``<root>/experiment_config.json``。若不存在则在 stderr
   打印错误并**不进行推送**（不回落到环境变量 remote，避免误推）。
3. 以上均未指定有效项目根文件时：可读环境变量 ``EXPERIMENT_KIT_GIT_CONFIG`` / ``COT_EVAL_GIT_CONFIG``
   指向的 JSON/YAML 文件。

字段优先级：关键字参数 > 配置文件 > 环境变量（remote/branch/subdir/push/experiment_name）。

推送目录：仓库内为 ``<experiment_name>/<subdir>/``（一级 experiment_name，二级 subdir；缺一则只拼已有项）。

环境变量（回退）：
  EXPERIMENT_KIT_GIT_REMOTE / COT_EVAL_RESULTS_GIT_REMOTE
  EXPERIMENT_KIT_GIT_BRANCH / COT_EVAL_RESULTS_GIT_BRANCH   默认 main
  EXPERIMENT_KIT_GIT_SUBDIR / COT_EVAL_RESULTS_GIT_SUBDIR
  EXPERIMENT_KIT_PUSH_GIT / COT_EVAL_PUSH_GIT   设为 0/false/no/off 禁用
  EXPERIMENT_KIT_GIT_CONFIG / COT_EVAL_GIT_CONFIG

``experiment_config.json`` 示例字段：remote、branch、subdir、push、experiment_name
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_LOG = "[experiment_kit]"

_COMMIT_MSG_MAX = 200

# #region agent log
_AGENT_DEBUG_LOG = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cursor", "debug-49fae6.log")
)


def _agent_debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: Dict[str, Any],
) -> None:
    try:
        import time

        payload = {
            "sessionId": "49fae6",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        os.makedirs(os.path.dirname(_AGENT_DEBUG_LOG), exist_ok=True)
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as df:
            df.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# #endregion


def _env_first(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default


def _parse_boolish(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if v == 0:
            return False
        if v == 1:
            return True
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("0", "false", "no", "off", ""):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
    return None


def _project_root_git_config_path(project_root: str) -> str:
    root = os.path.abspath(os.path.expanduser(project_root.strip()))
    return os.path.join(root, "experiment_config.json")


def resolve_project_git_push_config_path(project_root: str) -> Optional[str]:
    """
    若 ``<project_root>/experiment_config.json`` 存在则返回其绝对路径，否则 None。
    """
    if not project_root or not str(project_root).strip():
        return None
    p = _project_root_git_config_path(str(project_root))
    return p if os.path.isfile(p) else None


def _resolve_git_config_disk_path(
    git_config_path: Optional[str] = None,
    project_root_for_git_config: Optional[str] = None,
) -> Optional[str]:
    """返回应加载的配置文件绝对路径；project 模式下仅 <root>/experiment_config.json。"""
    if git_config_path and str(git_config_path).strip():
        ap = os.path.abspath(os.path.expanduser(str(git_config_path).strip()))
        if os.path.isfile(ap):
            return ap
        print(
            f"{_LOG} 错误: 指定的 Git 配置文件不存在: {ap}",
            file=sys.stderr,
        )
        return None
    if project_root_for_git_config and str(project_root_for_git_config).strip():
        p = _project_root_git_config_path(str(project_root_for_git_config))
        if os.path.isfile(p):
            return p
        return None
    env_path = _env_first("EXPERIMENT_KIT_GIT_CONFIG", "COT_EVAL_GIT_CONFIG")
    if env_path:
        ap = os.path.abspath(os.path.expanduser(env_path))
        if os.path.isfile(ap):
            return ap
        print(
            f"{_LOG} 错误: 环境变量指定的 Git 配置文件不存在: {ap}",
            file=sys.stderr,
        )
    return None


def _load_git_config_file(
    git_config_path: Optional[str] = None,
    project_root_for_git_config: Optional[str] = None,
) -> Dict[str, Any]:
    path = _resolve_git_config_disk_path(
        git_config_path=git_config_path,
        project_root_for_git_config=project_root_for_git_config,
    )
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        lp = path.lower()
        if lp.endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore
            except ImportError:
                print(
                    f"{_LOG} 读取 {path} 需要安装 PyYAML，已跳过",
                    file=sys.stderr,
                )
                return {}
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        return raw if isinstance(raw, dict) else {}
    except Exception as e:
        print(f"{_LOG} 读取 Git 配置失败 {path}: {e}", file=sys.stderr)
        return {}


def _push_enabled_from_env_only() -> bool:
    v = _env_first(
        "EXPERIMENT_KIT_PUSH_GIT",
        "COT_EVAL_PUSH_GIT",
        default="1",
    ).lower()
    return v not in ("0", "false", "no", "off")


def _resolve_push_flag(
    *,
    push_git_kw: Optional[bool],
    file_cfg: Dict[str, Any],
) -> bool:
    if push_git_kw is not None:
        return bool(push_git_kw)
    for key in ("push", "push_git", "push_enabled", "enabled"):
        if key in file_cfg:
            b = _parse_boolish(file_cfg.get(key))
            if b is not None:
                return b
    return _push_enabled_from_env_only()


def _str_from_cfg_or_kw(
    kw: Optional[str],
    file_cfg: Dict[str, Any],
    *cfg_keys: str,
) -> str:
    if kw is not None and str(kw).strip():
        return str(kw).strip()
    for k in cfg_keys:
        v = file_cfg.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _path_segments_for_remote_artifacts(experiment_name: str, subdir: str) -> List[str]:
    """
    远程产物目录：先 ``experiment_name`` 一级，再 ``subdir`` 二级（均支持用 ``/`` 写多级）。
    跳过空段及 ``.`` / ``..``，避免路径逃逸。
    """
    segments: List[str] = []
    for raw in (experiment_name or "").strip(), (subdir or "").strip():
        if not raw:
            continue
        for piece in raw.replace("\\", "/").split("/"):
            p = piece.strip()
            if not p or p == "." or p == "..":
                continue
            segments.append(p)
    return segments


def _push_log(msg: str, *, err: bool = False) -> None:
    """Git 推送流程日志，前缀 ``[experiment_kit] push:``。"""
    print(f"{_LOG} push: {msg}", file=sys.stderr if err else sys.stdout, flush=True)


def _commit_message(experiment_name: str, experiment_id: Optional[str]) -> str:
    name = (experiment_name or "").strip()
    eid = (experiment_id or "").strip() or "unknown"
    if name:
        msg = f"eval: {name} | id={eid}"
    else:
        msg = f"eval: {eid}"
    msg = msg.replace("\n", " ").replace("\r", " ").strip()
    if len(msg) > _COMMIT_MSG_MAX:
        msg = msg[: _COMMIT_MSG_MAX - 3] + "..."
    return msg


@dataclass
class GitPushSettings:
    """解析后的 Git 推送设置（remote 为空表示未配置、不应 push）。"""

    remote: str
    branch: str
    subdir: str
    push_enabled: bool
    experiment_name: str


def resolve_git_push_settings(
    *,
    git_remote: Optional[str] = None,
    git_branch: Optional[str] = None,
    git_subdir: Optional[str] = None,
    push_git: Optional[bool] = None,
    experiment_name: Optional[str] = None,
    git_config_path: Optional[str] = None,
    project_root_for_git_config: Optional[str] = None,
) -> GitPushSettings:
    """
    合并配置文件、环境变量与本次调用参数。

    若传入 ``project_root_for_git_config``，**必须**存在 ``<root>/experiment_config.json``，
    否则在 stderr 报错并返回空的 remote（不推送、不使用环境变量补 remote）。
    """
    pr = project_root_for_git_config
    if pr and str(pr).strip():
        req = _project_root_git_config_path(str(pr))
        if not os.path.isfile(req):
            print(
                f"{_LOG} 错误: 未找到 {req}。"
                f"请将 experiment_kit/experiment_config.example.json 复制到项目根并改名为 experiment_config.json 后填写。"
                f"已跳过 Git 推送。",
                file=sys.stderr,
            )
            return GitPushSettings(
                remote="",
                branch="main",
                subdir="",
                push_enabled=False,
                experiment_name="",
            )

    file_cfg = _load_git_config_file(
        git_config_path=git_config_path,
        project_root_for_git_config=project_root_for_git_config,
    )

    remote = (
        (git_remote or "").strip()
        or str(file_cfg.get("remote") or file_cfg.get("url") or "").strip()
        or _env_first("EXPERIMENT_KIT_GIT_REMOTE", "COT_EVAL_RESULTS_GIT_REMOTE")
    )
    branch = (
        (git_branch or "").strip()
        or str(file_cfg.get("branch") or "").strip()
        or _env_first(
            "EXPERIMENT_KIT_GIT_BRANCH",
            "COT_EVAL_RESULTS_GIT_BRANCH",
            default="main",
        )
        or "main"
    )
    subdir_raw = (
        (git_subdir or "").strip()
        or str(
            file_cfg.get("subdir")
            or file_cfg.get("git_subdir")
            or file_cfg.get("sub_dir")
            or ""
        ).strip()
        or _env_first("EXPERIMENT_KIT_GIT_SUBDIR", "COT_EVAL_RESULTS_GIT_SUBDIR")
    )
    subdir = subdir_raw.strip("/\\")

    exp_name = _str_from_cfg_or_kw(
        experiment_name,
        file_cfg,
        "experiment_name",
        "name",
        "experiment",
    )

    enabled = _resolve_push_flag(push_git_kw=push_git, file_cfg=file_cfg)
    return GitPushSettings(
        remote=remote,
        branch=branch,
        subdir=subdir,
        push_enabled=enabled,
        experiment_name=exp_name,
    )


def git_remote_url() -> str:
    return resolve_git_push_settings().remote


def git_branch() -> str:
    return resolve_git_push_settings().branch


def git_subdir() -> str:
    return resolve_git_push_settings().subdir


def should_push_eval_results(
    *,
    git_remote: Optional[str] = None,
    git_branch: Optional[str] = None,
    git_subdir: Optional[str] = None,
    push_git: Optional[bool] = None,
    experiment_name: Optional[str] = None,
    git_config_path: Optional[str] = None,
    project_root_for_git_config: Optional[str] = None,
) -> bool:
    s = resolve_git_push_settings(
        git_remote=git_remote,
        git_branch=git_branch,
        git_subdir=git_subdir,
        push_git=push_git,
        experiment_name=experiment_name,
        git_config_path=git_config_path,
        project_root_for_git_config=project_root_for_git_config,
    )
    return s.push_enabled and bool(s.remote)


def push_eval_artifacts_to_git(
    paths: List[str],
    *,
    experiment_id: Optional[str] = None,
    git_remote: Optional[str] = None,
    git_branch: Optional[str] = None,
    git_subdir: Optional[str] = None,
    push_git: Optional[bool] = None,
    experiment_name: Optional[str] = None,
    git_config_path: Optional[str] = None,
    project_root_for_git_config: Optional[str] = None,
) -> None:
    """
    clone（depth=1）-> 将 paths 拷入远程相对路径
    ``<experiment_name>/<subdir>/``（``experiment_name`` 一级、``subdir`` 二级；任一方为空则只拼另一方）；
    若二者皆空则拷至仓库根 -> add / commit / push。

    ``project_root_for_git_config``：子项目根目录，**仅**读取该目录下 ``experiment_config.json``；
    不存在则 stderr 报错且不推送。

    日志：各阶段输出 ``[experiment_kit] push: ...``（正常走 stdout，错误走 stderr），便于判断是否成功。
    """
    # #region agent log
    _agent_debug_log(
        "H2",
        "git_sync.py:push_eval_artifacts_to_git:entry",
        "push_eval_artifacts_to_git called",
        {
            "n_paths": len(paths),
            "experiment_id": experiment_id,
            "project_root_for_git_config": project_root_for_git_config,
        },
    )
    # #endregion
    settings = resolve_git_push_settings(
        git_remote=git_remote,
        git_branch=git_branch,
        git_subdir=git_subdir,
        push_git=push_git,
        experiment_name=experiment_name,
        git_config_path=git_config_path,
        project_root_for_git_config=project_root_for_git_config,
    )
    if not settings.push_enabled and not settings.remote:
        _push_log(
            "跳过（push 未启用且 remote 为空；若期望推送请检查 experiment_config.json 是否存在且已填写 remote、push）",
            err=True,
        )
        return
    if not settings.push_enabled:
        _push_log("跳过（配置中 push 已关闭）")
        return
    if not settings.remote:
        _push_log("跳过（remote 未配置）", err=True)
        return

    git_bin = shutil.which("git")
    if not git_bin:
        _push_log("中止：系统 PATH 中未找到 git", err=True)
        return

    remote = settings.remote
    branch = settings.branch

    abs_paths = [
        os.path.abspath(os.path.expanduser(p))
        for p in paths
        if isinstance(p, str) and p.strip() and os.path.exists(p)
    ]
    if not abs_paths:
        _push_log(
            f"跳过：paths 中无存在的路径（共传入 {len(paths)} 项）",
            err=True,
        )
        return

    work = tempfile.mkdtemp(prefix="experiment_kit_git_")
    clone_dir = os.path.join(work, "repo")
    try:
        exp_lbl = (settings.experiment_name or "").strip() or "(空)"
        sub_lbl = (settings.subdir or "").strip() or "(空)"
        _push_log(
            f"开始 | remote={remote!r} branch={branch!r} "
            f"experiment_name={exp_lbl!r} subdir={sub_lbl!r} "
            f"待上传路径数={len(abs_paths)}"
        )
        try:
            subprocess.run(
                [git_bin, "clone", "--depth", "1", remote, clone_dir],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            _push_log(
                f"git clone 失败:\n  {e.stderr or e.stdout or e}",
                err=True,
            )
            return

        _push_log(f"clone 完成（depth=1）-> {clone_dir}")
        co = subprocess.run(
            [git_bin, "-C", clone_dir, "checkout", branch],
            capture_output=True,
            text=True,
        )
        if co.returncode != 0:
            _push_log(
                f"git checkout {branch!r} 未成功，继续在克隆默认分支上操作:\n"
                f"  {co.stderr or co.stdout}",
                err=True,
            )
        else:
            _push_log(f"已检出分支 {branch!r}")

        rel_segments = _path_segments_for_remote_artifacts(
            settings.experiment_name,
            settings.subdir,
        )
        dest_root = (
            os.path.join(clone_dir, *rel_segments) if rel_segments else clone_dir
        )
        rel_in_repo = (
            "/".join(rel_segments) if rel_segments else "."
        )
        os.makedirs(dest_root, exist_ok=True)
        _push_log(f"仓库内目标目录（相对根）: {rel_in_repo!r}")

        for p in abs_paths:
            base = os.path.basename(p.rstrip(os.sep))
            target = os.path.join(dest_root, base)
            try:
                if os.path.isdir(p):
                    if os.path.exists(target):
                        shutil.rmtree(target)
                    shutil.copytree(p, target)
                else:
                    shutil.copy2(p, target)
            except OSError as e:
                _push_log(f"复制失败 {p} -> {target}: {e}", err=True)
                return

        copied = [os.path.basename(p.rstrip(os.sep)) for p in abs_paths]
        _push_log(f"已复制 {len(copied)} 项 -> {rel_in_repo!r}: {', '.join(copied)}")
        rel_add = os.path.relpath(dest_root, clone_dir)
        add_arg = "." if rel_add in (".", "") else rel_add

        subprocess.run(
            [git_bin, "-C", clone_dir, "add", "--", add_arg],
            check=True,
            capture_output=True,
            text=True,
        )
        _push_log(f"git add {add_arg!r}")

        st = subprocess.run(
            [git_bin, "-C", clone_dir, "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        if not (st.stdout or "").strip():
            _push_log("无变更可提交，跳过 commit/push", err=True)
            return

        msg = _commit_message(settings.experiment_name, experiment_id)
        _push_log(f"git commit 说明: {msg!r}")
        try:
            subprocess.run(
                [git_bin, "-C", clone_dir, "commit", "-m", msg],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            _push_log(
                f"git commit 失败:\n  {e.stderr or e.stdout or e}",
                err=True,
            )
            return

        try:
            subprocess.run(
                [git_bin, "-C", clone_dir, "push", "origin", branch],
                check=True,
                capture_output=True,
                text=True,
            )
            _push_log(
                f"成功 | git push origin {branch!r} | 远程子路径 {rel_in_repo!r} | remote={remote!r}"
            )
        except subprocess.CalledProcessError as e:
            _push_log(
                f"git push 失败:\n  {e.stderr or e.stdout or e}",
                err=True,
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)
