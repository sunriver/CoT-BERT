# CoT-BERT 实验结果可视化看板

FastAPI 读取本地 `eval_results` / `cot_bert_eval` 等目录下的 JSON，Vue3 提供运行列表、详情与多 run 对比（类 Neptune 的轻量版）。

## 数据目录

默认扫描（相对 `code-for-BERTNew2`）：

1. `eval_results/`
2. `../../expirments/bertNew2/cot_bert_eval`（与 CoT-BERT 同级的 `expirments`，若存在）
3. `eval_results/gold_cosine_scatter/`（若存在）

此外会索引 **`gold_cosine_scatter_*.png`**（按 `sick_test` / `sts_test` 文件名解析 run id），Gold–Cosine 页在仅有 PNG、无 `run_summary` 时也会列出 run。

通过环境变量覆盖（逗号分隔，支持相对 `code-for-BERTNew2` 的路径）：

```bash
export EXPERIMENT_VIZ_DATA_DIRS="/path/to/eval_results,/path/to/cot_bert_eval"
```

刷新内存索引（不重启进程）：

```bash
curl -X POST http://127.0.0.1:8000/api/refresh
```

## 后端

```bash
cd experiment_dashboard/backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- API：`http://127.0.0.1:8000/api/health`
- 文档：`http://127.0.0.1:8000/docs`

CORS 默认允许 `http://127.0.0.1:5173`；可设置：

```bash
export EXPERIMENT_VIZ_CORS_ORIGINS="http://127.0.0.1:5173,http://localhost:3000"
```

## 前端（开发）

```bash
cd experiment_dashboard/frontend
npm install
npm run dev
```

浏览器打开 Vite 提示的地址（一般为 `http://127.0.0.1:5173`），`/api` 由 Vite 代理到8000。

## 单端口部署（可选）

```bash
cd experiment_dashboard/frontend && npm run build
mkdir -p ../backend/static
cp -r dist/* ../backend/static/
cd ../backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

若 `backend/static/index.html` 存在，FastAPI 会在挂载 API 之后提供静态站点；前端路由使用 history模式，由 `StaticFiles(..., html=True)` 回退到 `index.html`。

## API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查与数据目录 |
| POST | `/api/refresh` | 重新扫描 JSON |
| GET | `/api/runs?q=&tab=&limit=` | 运行列表；`tab=senteval\|alignment\|gold\|all` 对应三类评估产物 |
| GET | `/api/runs/{run_id}` | 详情（超参、指标、artifact列表） |
| GET | `/api/compare?run_ids=a,b,c` | 指标与超参差异 |
| GET | `/api/runs/{run_id}/files/{filename}` | 安全下载/展示该 run 已索引文件 |

## 关联文件类型

- `eval_test_*.json`：主日志（SentEval、`train_config_full`）
- `run_summary_*.json`：金标散点摘要与图路径
- `alignment_uniformity_benchmark*.json`：Alignment / Uniformity

同一 `eval_run_tag`（或文件名后缀）的记录会在详情中合并展示。
