# experiments/ — 历史实验数据

Phase 3–5 的历史实验记录。这些实验使用**旧协议**（预提取特征、不同的数据划分），**不可与 AC 套件 (`results/logs/`) 的投稿数值直接对比**。

## 当前有效文件

| 文件 | 阶段 | 内容 |
|------|------|------|
| `phase3_ablation.json` | Phase 3 | 模块消融实验结果（60/20/20 划分，含 A1/A2/A2b/A3/B2/B3） |
| `cross_language_results.json` | Phase 4 | 跨语言迁移：EN→TE (19.68%), TE→EN (28.15%) |
| `v5_622/` | Phase 5 | 最终 6:2:2 协议消融矩阵（详见该目录下的 README.md） |

## last/ — 归档

| 内容 | 说明 |
|------|------|
| `runs/` | Phase 3 单次实验的 config + metrics + results（6 个 run） |
| `logs/` | 旧实验日志 |
| `cloud_fixed_results.json` | 云端修复后结果快照 |
| `model_comparison_results.json` | emotion2vec vs WavLM 对比 |
| `phase3_stdout.log` | 训练终端输出 |
| `phase4_record.md` | Phase 4 工作记录 |
| `registry.csv` | 旧实验索引 |

## v5_622/ 目录说明

Phase 5 最终协议（外层 7:3 + 内层 val → 有效 6:2:2）的完整实验数据。

| 文件 | 内容 |
|------|------|
| `v5_results.json` / `v5_save_results.json` | Run 1 / Run 2 的完整消融结果（A1/A2b/A3/B3/C1-C4, 3 seeds） |
| `A3_results.json` | A3（Prosody-only, 无 Adapter）独立运行结果 |
| `unified_fd_results.json` | FD 值汇总：Age=6.87, C1=0, C2=9.87, C3=8.71, C4=11.99, Lang=137.40 |
| `fd_lang_result.json` | 跨语言 FD：EN vs TE = 16.48（从真实 WavLM 特征计算） |
| `backbone_comparison.json` | WavLM vs Wav2Vec2 主干对比 |
| `mfcc_baseline_result.json` | MFCC 基线：WA=50.18% |
| `iemocap_contrast_result.json` | IEMOCAP 成人对照：A1=54.50%, A3=52.38% |
| `cremad_contrast_result.json` | CREMA-D 成人对照：A1=64.69%, A3=65.44% |
| `a2_stat_prior_results.json` | Adapter 统计先验初始化消融 |
| `statistical_tests.json` | Bootstrap 显著性检验（C-BESD A1→A3: +2.23pp, 95% CI [1.41, 3.64]） |
| `supplementary_results.json` | 补充实验 |
| `attention_analysis/` | 每类注意力统计（entropy/标准差/准确率） |
| `fd_vs_accuracy_unified.png` | FD vs Accuracy 统一框架图 |
| `run_v5.log` / `run_v5_save.log` | 终端输出日志 |
