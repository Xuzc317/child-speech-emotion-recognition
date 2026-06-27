# 项目进度日志

## 2026-06-16 — 全部实验完成 & 项目大整理

### 实验完成
- [x] B1-B7 全部 192/192 完成
- [x] 本地 + 云端双检通过 (`verify_all_192.py`)

### 工作区整理
- [x] `results/` — 整合 results_remote，B1-B7 192 JSONs + analysis/ + archive/
- [x] `checkpoints/` — B1-B7 权重整理，旧权重归档到 archive/
- [x] `docs/` — 旧文档归档，ops/ 重写
- [x] `paper_draft/` — 重组为 current/archive/presentations + figures
- [x] `experiments/` — 旧协议数据归档
- [x] `scripts/` — 新增 verify_all_192.py + download_checkpoints.py
- [x] `.gitignore` — 全面更新，排除大文件和密钥
- [x] Git 历史清理 — 移除 >50MB blob，成功 push
- [x] 根目录 — 删除 AGENTS.md, task_plan.md，更新所有 README
- [x] `requirements.txt` — 补全依赖
- [x] 云端权重下载 — B5/B6/B7 + B4补缺 (64 files, ~23 GB)

### 下一步
- [ ] 论文打磨
- [ ] B1-B7 混淆矩阵生成（见 `results/TODO_补充清单.md`）
- [ ] XAI 更新（基于 B1/B6 最优模型）
- [ ] 投稿准备
