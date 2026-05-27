# INTERSPEECH / ICASSP 投稿前检查清单

## 文稿

- [x] 摘要 / 引言 / 方法 / 实验 / 讨论 / 结论 LaTeX（`paper_draft/`）
- [x] `main.tex` + `references.bib`
- [x] 英文 / 中文 Word 初稿
- [ ] 替换为官方 INTERSPEECH 模板（`main.tex` documentclass）
- [ ] 作者信息与致谢（投稿前去匿名）

## 实验可复现

- [x] 6 组 JSON 与论文矩阵一致（`python scripts/verify_experiment_jsons.py`）
- [x] `publication_package/experiment_results.csv`
- [ ] 远端或本地 checkpoint（`checkpoints/autodl/`）
- [ ] 混淆矩阵图（需 checkpoint：`scripts/plot_confusion_matrix.py`）
- [ ] Layer 8 权重图（`scripts/export_layer_weights.py`）

## 配图

- [x] fig00 架构图
- [x] fig01–fig05 实验与 XAI
- [ ] fig06+ 混淆矩阵（待权重）
- [ ] fig_layer_weights（待 checkpoint）

## 打包分享

```bash
python scripts/build_submission_bundle.py
```

输出目录：`submission_bundle/`（LaTeX + 图 + publication_package + docx）

## AutoDL 恢复（可选）

```bash
python scripts/tmp_paramiko_autodl_runner.py --probe
python scripts/tmp_paramiko_autodl_runner.py --sync-checkpoints
python scripts/export_layer_weights.py --checkpoint checkpoints/autodl/exp2_prosody_guided/best_model.pt
```

若 `--probe` 失败，实例已释放，需重新开机器训练或接受当前 JSON+XAI 投稿包。
