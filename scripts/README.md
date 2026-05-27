# scripts/ — 脚本工具集

所有实验管理、数据处理、评估和可视化的脚本。按功能分组说明。

## AC 套件核心（投稿使用）

| 脚本 | 用途 |
|------|------|
| `verify_experiment_jsons.py` | **权威验收脚本**：检查 12 个 JSON 与 CANONICAL 值一致 → `python scripts/verify_experiment_jsons.py` |
| `autodl_ac_suite.sh` | AutoDL 云端完整 AC 套件（A1 训练 + A2 CM + C1/C2） |
| `autodl_ac_suite_resume.sh` | 云端 resume（断点续跑 CM + C1 + C2） |
| `tmp_paramiko_autodl_runner.py` | Paramiko SSH 自动化：pull/push/train/status |
| `finish_remote_c2.py` | 云端 C2 补跑（layer_weights + APC + XAI） |
| `_wait_resume.py` | 轮询等待云端任务完成 |

## 本地后处理（AutoDL 完成后执行）

| 脚本 | 用途 |
|------|------|
| `merge_remote_logs.py` | `results_remote/` → `results/logs/` + `publication_package/` 合并 |
| `aggregate_fau_multiseed.py` | FAU 多 seed 汇总 → `fau_multiseed_summary.json` |
| `sync_autodl_canonical_logs.py` | 写入 canonical JSON |
| `run_post_autodl_pipeline.py` | 本地 post 全流程（merge + verify + figures + docx + bundle） |
| `run_local_checklist_jobs.py` | 本地验收（非训练）：verify + diagnostics + figures |

## 诊断与分析（C2）

| 脚本 | 用途 |
|------|------|
| `run_prosody_diagnostics.py` | C2 诊断主脚本：layer_weights + APC + XAI |
| `extract_diagnostics.py` | 底层诊断提取（被 `run_prosody_diagnostics.py` 调用） |
| `export_layer_weights.py` | 单独导出层权重 |
| `compute_canonical_fd.py` | **2026-05-27 新增**：在 AutoDL GPU 上计算 canonical FD/SMMD |
| `measure_distribution_shift.py` | 旧版 FD 测量（增强实验用） |

## 论文出图

| 脚本 | 用途 |
|------|------|
| `generate_paper_figures.py` | 生成 fig01–fig05 |
| `generate_architecture_figure.py` | 生成 fig00 系统架构图 |
| `plot_confusion_matrix.py` | 从 checkpoint 画混淆矩阵（fig07/08, figA1–A4） |
| `plot_layer_weights.py` | 画 fig06 层融合权重图 |
| `plot_fd_vs_accuracy.py` | 画 FD vs Accuracy 图 |

## 文档导出

| 脚本 | 用途 |
|------|------|
| `generate_docx.py` | 英文 Word 初稿 |
| `generate_docx_cn.py` | 中文 Word 初稿 |
| `generate_v5_docx.py` | v5 版 Word（旧） |
| `build_submission_bundle.py` | 构建投稿打包目录 `submission_bundle/` |
| `export_data_doc.py` | 导出数据文档 |

## 特征提取（Phase 1–2，旧协议）

| 脚本 | 用途 |
|------|------|
| `extract_ssl_features.py` | 预提取 WavLM/emotion2vec SSL 特征 |
| `extract_prosody_features.py` | 预提取韵律特征（F0 + energy） |
| `extract_augmented_features.py` | 离线预增强 + SSL 特征提取 |
| `compute_adapter_init.py` | Adapter 统计先验初始化计算 |

## 历史实验（Phase 3–5，旧协议）

| 脚本 | 用途 |
|------|------|
| `run_experiments_v5.py` | v5 6:2:2 协议消融实验（第一轮） |
| `run_cloud_all.py` | 云端全量实验 |
| `run_cross_language.py` | 跨语言迁移实验 |
| `run_iemocap_contrast.py` | IEMOCAP 成人对照 |
| `run_cremad_contrast.py` | CREMA-D 成人对照 |
| `run_mfcc_baseline.py` | MFCC 基线 |
| `run_backbone_comparison.py` | WavLM vs Wav2Vec2 对比 |
| `run_e2v_large.py` | emotion2vec+ large 模型测试 |
| `run_a2_stat_prior.py` | A2 统计先验消融 |
| `run_supplementary_experiments.py` | 补充实验 |
| `run_model_comparison.py` | 模型对比 |
| `compute_statistical_tests.py` | Bootstrap 显著性检验 |
| `compute_fd_lang.py` | 跨语言 FD 计算 |
| `unified_fd_analysis.py` | 统一 FD 分析 |
| `analyze_attention.py` | 注意力统计分析 |
| `analyze_besd_final.py` | BESD 最终分析 |

## 工具与辅助

| 脚本 | 用途 |
|------|------|
| `generate_paper.py` | 旧版论文生成 |
| `generate_ppt.py` | 旧版 PPT 生成 |
| `evaluate_model.py` | 模型评估 |
| `gen_confusion_matrix.py` | 旧版混淆矩阵 |
| `regenerate_features.py` | 重新提特征 |
| `check_cloud_status.py` | 云服务器状态检查 |
| `poll_and_download.py` | 轮询并下载结果 |
| `download_models.py` | 下载模型权重 |
| `reevaluate_exp4_test_split.py` | **2026-05-27 新增**：Exp4 test-only 重评估 |

## last/

`run_experiments.sh`：旧版实验启动 Shell 脚本。
