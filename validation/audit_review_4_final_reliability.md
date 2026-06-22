# 最终数据可靠性审计

> 审计日期: 2026-06-22 | 模式: 短上下文防爆（脚本遍历，仅读小结）

## Step 0 — 项目最大文件清单（确认规避对象）

| 大小 | 路径 |
|------|------|
| 364.0 MB | checkpoints/autodl/b6/E6-05_s456/best_model.pt |
| 364.0 MB | checkpoints/autodl/b6/E6-05_s42/best_model.pt |
| 364.0 MB | checkpoints/autodl/b6/E6-05_s123/best_model.pt |
| 364.0 MB | checkpoints/autodl/b6/E6-10_s456/best_model.pt |
| 364.0 MB | checkpoints/autodl/b6/E6-10_s42/best_model.pt |
| 364.0 MB | checkpoints/autodl/b6/E6-10_s123/best_model.pt |
| 363.6 MB | checkpoints/autodl/b6/E6-02_s456/best_model.pt |
| 363.6 MB | checkpoints/autodl/b6/E6-02_s42/best_model.pt |
| 363.6 MB | checkpoints/autodl/b6/E6-02_s123/best_model.pt |
| 363.6 MB | checkpoints/autodl/b6/E6-07_s456/best_model.pt |

全部为 `.pt` checkpoint（150+ 个，b1-b7 全目录），**全程不读取其内容**。

次大非checkpoint文件：presentation_v4.pptx (30MB), _legacy/*.npy (28MB×2，旧MFCC特征), PDF/PNG 配图/参考文献 — 均与本次指标审计无关，不读取。

`validation/` 下报告类文件实测均 <25KB（比预期小），仅 `all_tracked_files.txt` 为 79.4KB，按规则 >50KB 不整读，仍用 grep/Python 取行。

## 审计项 1 — 一键可复现性

**结果: PASS**

证据：
- `python scripts/verify_all.py` → `Passed: 4/4, Failed: 0/4`（gen_manifest.py / regen_handbook.py / check_metrics.py / phase4_audit.py 全 PASS）
- 额外单独重跑 `regen_handbook.py`（生成194行手册）+ `rebuild_manifest.py`（生成100行+表头，3条INVALID: E4-04/E1-08/E4-10）
- `git status --short` 重跑后仅有 `validation/FINAL_RELIABILITY_CHECK.md`（本审计报告自身，新建）为未跟踪文件，**无任何其他文件被改动** → 已提交版 = 重生版，可复现性确认。

---

## 审计项 2 — 无 checkpoint 入库

**结果: PASS**

证据：
- `git ls-files` 共 1358 个跟踪文件，按后缀过滤 `.pt/.ckpt/.bin/.pth/.safetensors` → 命中 **0**
- 扩展核查（.h5/.onnx/.npz/.msgpack/.tar/.pkl 等）→ 命中 7，均为 `.npz`（非模型权重格式，数据/特征缓存类，不在禁止清单内，不视为违规）
- `.gitignore` 中确认存在 checkpoints 相关排除规则

---

## 审计项 3 — 负荷数字独立复核

**结果: PASS**

独立脚本仅打开 7 个实验 × 3 seed = 21 个 JSON（均 <1KB），现算 mean(ddof=1)：

| exp   | wa_mean | wa_std | uar_mean | uar_std | 手册对照 |
|-------|---------|--------|----------|---------|---------|
| E1-02 | 91.87   | 1.56   | 91.85    | 1.54    | 91.87% ✓ |
| E1-05 | 67.05   | 0.67   | 42.92    | 1.85    | 67.05% ✓ |
| E2-01 | 96.91   | 0.19   | 96.88    | 0.21    | 96.91% ✓ |
| E6-03 | 91.91   | 0.94   | 91.86    | 0.92    | 91.91% ✓ |
| E6-04 | 91.96   | 0.93   | 91.91    | 0.93    | 91.96% ✓ |
| E7-03 | 91.57   | 0.45   | 91.52    | 0.45    | 91.57±0.44% ✓(±0.01舍入) |
| E7-05 | 91.17   | 1.28   | 91.14    | 1.28    | 91.17±1.28% ✓ |

7/7 WA 均值与手册数字逐位一致（E7-03 std 差0.01pp系舍入）。E1-05 UAR(42.92) 略低于手册"in-domain UAR"42.95%——差异来自统计口径（单实验 vs 多实验聚合），留待审计项4专门核验。

---

## 审计项 4 — FAU UAR 两口径

**结果: PASS**

train_data/test_data 字段为列表，首元素=主语料、第二元素(若有)=混入的外域增强数据（如 E4-06/E4-08 主体是FAU混入IEMOCAP）。按此口径独立实现：
- in-domain（train==test 且主语料=fau-aibo，剔除3条INVALID）= **42.95±3.99%** (n=63) ✓ = 手册值
- by_test_data（test主语料=fau-aibo，剔除3条INVALID）= **41.72±5.72%** (n=69) ✓ = 手册值

副产品：核查发现 E1-08/E4-04/E4-10 均不落在 FAU 主语料集合中（INVALID剔除对此二指标实为no-op），且亲眼验证 E4-10 三个seed配置确实不一致（s123/s42=`['iemocap']`纯净，s456=`['iemocap','fau-aibo']`混入），与手册"跨seed配置不一致"说法一致，构成审计项5的旁证。

---

## 审计项 5 — INVALID 落地

**结果: PASS**

- `validation/provenance_manifest.csv`（100行）中 `aggregation_valid=='FALSE'` 精确计数 = **3**，对应 `['E1-08', 'E4-04', 'E4-10']`，与手册一致；其余97条均为 TRUE。
- 该CSV虽标记FALSE，但仍保留原始数值（`test_wa_mean+-std`等列）供溯源排查，非违规——真正面向论文的权威手册才是"排除"判定的落地位置。
- `docs/current/权威数据手册.md` 逐行核查：
  - line 8: `> INVALID experiments EXCLUDED from all aggregations: E1-08, E4-04, E4-10`
  - line 40/100/106: 三条经验的聚合单元格逐字打印为 `INVALID | INVALID`（WA列和UAR列），**不是数值**，与原始per-seed值（如E1-08的63.21%/64.25%/63.82%）并列展示但聚合列明确置为INVALID文本
  - line 46/112/113: 各自给出排除原因（与审计项4副产品中亲自验证的E4-10跨seed配置不一致完全对应）
- 计数=3 ✓，聚合单元格=INVALID文本而非数值 ✓，两项均满足用户要求。

---

## 审计项 6 — 无残留凭空值

**结果: PASS**

逐一 grep 手册全文，4 个旧/可疑数字命中且仅命中 1 次：

| 数字 | 命中数 | 上下文 |
|------|--------|--------|
| 41.05% | 1 | L89: "Previous handbook values (41.05%, 35.47%, 19.17% as range minimum) are **PHANTOM** - not found in any log." |
| 35.47% | 1 | 同上（同一行） |
| 19.17% | 1 | 同上（同一行） |
| 76.02% | 1 | L60: "The previous claim 'FAU +8.2pp' was based on a **phantom** 76.02% value not found in any log." |

四个数字均仅出现在明确标注"PHANTOM/phantom"（凭空值）的纠错说明行中，未在任何数据表格或结论性陈述中以有效数值形式出现 → 无残留污染。

---

## 审计项 7 — 信任边界声明

**结果: PASS**

- `validation/audit_00_SUMMARY.md` L33: "**WA/UAR cannot be recomputed** — 0/192 have raw predictions or confusion matrices"
- `docs/current/权威数据手册.md` L224: "**Predictions**: 0/192 files contain predictions/confusion matrices - WA/UAR trusted as-is from sklearn"

两份文档均明确声明：192个日志均不含原始预测/混淆矩阵，WA/UAR 数值本身无法独立重算，只能取信训练时 sklearn 的计算结果（即取信训练代码）。该信任边界已书面记录，非隐藏假设。

---

## 总体结论

**RELIABLE**

7/7 审计项全部 PASS：
1. 一键可复现性 ✓ 2. 无checkpoint入库 ✓ 3. 负荷数字独立复核(7实验×21日志逐位匹配) ✓
4. FAU UAR两口径(42.95%/41.72%精确匹配) ✓ 5. INVALID落地(计数=3且聚合显式置INVALID) ✓
6. 无残留凭空值(4个旧数字均只在PHANTOM纠错行出现) ✓ 7. 信任边界已书面声明 ✓

**已知边界（非缺陷，文档已自述）**：
- WA/UAR 为训练时 sklearn 计算结果，因无原始预测/混淆矩阵存档，无法从0开始独立重算，只能验证"复现一致性"而非"绝对正确性"
- B7 源域 checkpoint 未在日志中独立记录，依赖 `launch_b7.sh` 执行正确性
- E1-08/E4-04/E4-10 因跨seed配置不一致已被手册和manifest一致排除，不计入任何聚合结论

未发现新增问题。本次审计未修改/提交除本报告外的任何文件。
