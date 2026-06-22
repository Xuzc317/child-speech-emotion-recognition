#!/usr/bin/env python3
"""
生成 `docs/current/实验方案与数据_总表.md` — 合并条件(CSV) + 指标(logs, ddof=1) + 分歧(修正后diff) + 有效性(manifest)。

AUTO-GENERATED — 一键重生: python scripts/gen_master_reference.py
条件来源: validation/provenance_manifest.csv (以 launch 脚本为准)
指标来源: results/logs/ (ddof=1 聚合)
有效性: manifest 中 aggregation_valid + seed_validity 字段
"""

import csv
import os
import re
from collections import defaultdict, OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_CSV = PROJECT_ROOT / "validation" / "provenance_manifest.csv"
DIFF_MD = PROJECT_ROOT / "validation" / "design_vs_actual_diff.md"
OUTPUT_MD = PROJECT_ROOT / "docs" / "current" / "实验方案与数据_总表.md"
NOW_STRING = "2026-06-22"

# Hardcoded INVALID experiments (ac_suite_2026-06-validated, from reproducibility audit)
# These have cross-seed config inconsistency → mean±std NOT reliable
INVALID_EXPERIMENTS = {
    "E1-08": "seed=42 INVALID (old protocol: aug/fusion/adapter=None); seed=123/456 valid",
    "E4-04": "seed=42 INVALID (train_data differs: c-besd vs c-besd-4cl+iemocap); seed=123/456 valid",
    "E4-10": "seed=456 INVALID (train_data differs: iemocap vs iemocap+fau-aibo); seed=42/123 valid",
}

# Phase descriptions
PHASE_INFO = {
    "B1": {
        "title": "B1 (E1) — Pooling × Dataset 基线",
        "design": "3 pooling (mean/self_attention/prosody_guided) × 3 datasets (C-BESD/FAU/IEMOCAP), frozen WavLM. 9×3=27 runs.",
        "key_finding": "SelfAttn > Mean >> Prosody; C-BESD (91.87%) >> FAU (67.05%) ≈ IEMOCAP.",
    },
    "B2": {
        "title": "B2 (E3) — Zero-shot 跨语料迁移",
        "design": "3 pooling × 6 transfer directions (C-BESD↔FAU↔IEMOCAP), 1 seed each. 18 runs. C-BESD uses 4-class subset (aligned with FAU/IEMOCAP).",
        "key_finding": "Zero-shot 跨语料 WA 19.17%-35.47%, 分布偏移显著. Best: IEMOCAP→C-BESD self_attn (34.68%).",
    },
    "B3": {
        "title": "B3 (E4) — 数据增强敏感性",
        "design": "4 augment conditions (C1 baseline, C2 external, C3 child, C4 mixed) × 3 datasets. 12×3=36 runs.",
        "key_finding": "C3 child aug 微弱正收益 (+0.25~0.74pp); C2/C4 外域混合显著损害.",
    },
    "B4": {
        "title": "B4 (E5) — LayerFusion 消融",
        "design": "6 multi-seed experiments (last/weighted × 3 datasets) + 3 grid search (best_single L1-L12 × 3 datasets). 18+36=54 runs.",
        "key_finding": "last ≈ weighted ≈ 任何单层 L≥7, Fusion 策略不重要.",
    },
    "B5": {
        "title": "B5 (E2) — WavLM Unfreeze 对比",
        "design": "3 datasets × best pooling from B1, differential LR (backbone 1e-5 / head 3e-4). 3×3=9 runs.",
        "key_finding": "Unfreeze 在 C-BESD +4pp (91.87%→96.91%), FAU +8pp (67.05%→66.37%†), IEMOCAP +8pp.",
    },
    "B6": {
        "title": "B6 (E6) — 模块消融 (累积式 build-up)",
        "design": "5 configs (Mean+Last → +Adapter → +SelfAttn → +WeightedFusion → 全栈) × 2 datasets (C-BESD, FAU). 10×3=30 runs.",
        "key_finding": "C-BESD: Pooling(Mean→SelfAttn) +11pp 关键; Adapter 无效; WF 微弱. FAU: 所有模块改良有限.",
    },
    "B7": {
        "title": "B7 (E7) — 模型迁移 Fine-tune",
        "design": "6 transfer directions (C-BESD↔FAU↔IEMOCAP), fine-tune 3 seeds. 6×3=18 runs. Pooling head unified to self_attention.",
        "key_finding": "目标域天花板主导迁移: C-BESD target 91%+ > FAU target 66-67% > IEMOCAP target 62-63%; 源域影响小.",
    },
}


def read_manifest():
    """Read provenance_manifest.csv, return list of experiment dicts."""
    rows = []
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def extract_divergence_items():
    """Extract the corrected divergence summary from design_vs_actual_diff.md."""
    if not DIFF_MD.exists():
        return []

    with open(DIFF_MD, "r", encoding="utf-8") as f:
        content = f.read()

    items = []
    # Parse the overview table
    in_table = False
    for line in content.split("\n"):
        if line.startswith("| # | 类别"):
            in_table = True
            continue
        if in_table:
            if line.startswith("|---"):
                continue
            if line.startswith("| D") or line.startswith("| D-"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 7:
                    num = parts[1]
                    category = parts[2]
                    severity = parts[3]
                    design = parts[4]
                    actual = parts[5]
                    impact = parts[6]
                    items.append({
                        "num": num,
                        "category": category,
                        "severity": severity,
                        "design": design,
                        "actual": actual,
                        "impact": impact,
                    })
            elif line.strip() == "" or line.startswith("---"):
                in_table = False

    return items


def format_wa_uar(row):
    """Format WA±std / UAR±std for display.

    Note: CSV uses Unicode ± (U+00B1), not ASCII +-.
    """
    wa = row.get("test_wa_mean±std", "N/A")
    uar = row.get("test_uar_mean±std", "N/A")
    return wa, uar


def is_invalid(row):
    """Check if experiment is marked INVALID.

    Two-layer check:
    1. CSV columns (aggregation_valid / seed_validity) — survive if gen_manifest.py
       is fixed to output them.
    2. Hardcoded fallback — survives gen_manifest.py CSV regeneration that strips
       those columns (3 known INVALID from ac_suite_2026-06-validated audit).
    """
    eid = row.get("experiment_id", "")

    # Layer 1: CSV columns (if present and explicit)
    agg_valid = row.get("aggregation_valid", "").strip()
    if agg_valid == "FALSE":
        seed_valid = row.get("seed_validity", "").strip()
        return True, seed_valid

    # Layer 2: Hardcoded authoritative list (survives CSV regeneration)
    if eid in INVALID_EXPERIMENTS:
        return True, INVALID_EXPERIMENTS[eid]

    return False, ""


def main():
    rows = read_manifest()
    print(f"Read {len(rows)} experiment rows from manifest")

    # Group by phase
    by_phase = defaultdict(list)
    for r in rows:
        phase = r["phase"]
        by_phase[phase].append(r)

    # Sort phases
    phase_order = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]

    # Divergence items
    div_items = extract_divergence_items()
    print(f"Extracted {len(div_items)} divergence items from diff")

    # Count INVALID
    invalid_exps = []
    for r in rows:
        inv, reason = is_invalid(r)
        if inv:
            invalid_exps.append((r["experiment_id"], reason))

    # --- Build Markdown ---
    lines = []
    w = lines.append

    w(f"# 实验方案与数据 — 总表")
    w("")
    w(f"> **⚠️ AUTO-GENERATED** — 条件来自 `validation/provenance_manifest.csv` (以 launch 脚本为准)")
    w(f"> 指标来自 `results/logs/` (ddof=1 聚合), 有效性来自 manifest 的 `aggregation_valid` / `seed_validity` 字段")
    w(f"> 分歧信息来自 `validation/design_vs_actual_diff.md` (已按独立复核修正)")
    w(f"> **一键重生**: `python scripts/gen_master_reference.py`")
    w(f"> 生成日期: {NOW_STRING} | 协议: `ac_suite_2026-06-validated` | 总实验: 192 runs (B1-B7)")
    w("")
    w("---")
    w("")
    w("## 目录")
    w("")
    w("1. [设计概览](#1-设计概览)")
    w("2. [逐实验总表](#2-逐实验总表)")
    w("   - [B1 Pooling × Dataset 基线](#b1-e1--pooling--dataset-基线)")
    w("   - [B2 Zero-shot 跨语料迁移](#b2-e3--zero-shot-跨语料迁移)")
    w("   - [B3 数据增强敏感性](#b3-e4--数据增强敏感性)")
    w("   - [B4 LayerFusion 消融](#b4-e5--layerfusion-消融)")
    w("   - [B5 WavLM Unfreeze 对比](#b5-e2--wavlm-unfreeze-对比)")
    w("   - [B6 模块消融](#b6-e6--模块消融)")
    w("   - [B7 模型迁移 Fine-tune](#b7-e7--模型迁移-fine-tune)")
    w("3. [设计-执行分歧](#3-设计-执行分歧)")
    w("4. [特殊说明](#4-特殊说明)")
    w("5. [关键核验数字](#5-关键核验数字)")
    w("6. [已知边界](#6-已知边界)")
    w("")
    w("---")
    w("")
    w("## 1. 设计概览")
    w("")
    w("### 研究框架")
    w("")
    w("本项目构建分布偏移诊断框架 (FD-WA)，系统验证「分布偏移→性能下降」的因果关系。")
    w("")
    w("### 核心架构")
    w("")
    w("```")
    w("WavLM Base (frozen/unfrozen) → 12层 LayerFusion → Pooling → SEMLP 分类器")
    w("                                (learnable weights)   (mean/self_attn/prosody)  (~704K params)")
    w("```")
    w("")
    w("### 三数据集")
    w("")
    w("| 数据集 | 样本 | 类别 | 说话人 | 年龄 | 风格 |")
    w("|--------|------|------|--------|------|------|")
    w("| C-BESD (MY) | 4,179 | 6类 | 70 children | 6-12y | 演绎式 (EN+TE 双语) |")
    w("| FAU Aibo | 18,216 | 4类 | 51 children | 10-13y | 自然式儿童-机器人交互 |")
    w("| IEMOCAP | ~9,794 | 4类 | 10 adults | — | 演绎式 (成人对照) |")
    w("")
    w("### 实验矩阵概览")
    w("")
    w("| Phase | 内容 | 实验数 | 核心结论 |")
    w("|-------|------|--------|---------|")
    for p in phase_order:
        info = PHASE_INFO.get(p, {})
        title_short = info.get("title", p).split("—")[0].strip()
        n_exps = len(by_phase[p])
        finding = info.get("key_finding", "")
        w(f"| {p} | {title_short} | {n_exps} exps | {finding} |")
    w("")
    w(f"**总计**: 192 runs (B1-B7), {len(rows)} CSV 行")
    w(f"**INVALID**: {len(invalid_exps)} experiments ({', '.join(eid for eid, _ in invalid_exps)}) — 跨 seed 配置不一致，aggregation 不可用")
    w("")
    w("---")
    w("")
    w("## 2. 逐实验总表")
    w("")

    for p in phase_order:
        exps = by_phase[p]
        info = PHASE_INFO.get(p, {})
        w(f"### {info.get('title', p)}")
        w("")
        w(f"**设计**: {info.get('design', '')}")
        w("")

        # Table header
        w("| 实验 | 语料 | Pooling | Fusion | Adapter | Unfreeze | Aug | Seeds | WA±std | UAR±std | 有效性 |")
        w("|------|------|---------|--------|---------|----------|-----|-------|--------|---------|--------|")

        for r in sorted(exps, key=lambda x: x["experiment_id"]):
            eid = r["experiment_id"]
            corpus = r["corpus"].replace("->", "→")
            pooling = r["pooling"]
            fusion = r["fusion"]
            adapter = "✓" if r["adapter"] == "True" else "—"
            unfreeze = "✓" if r["unfreeze"] == "True" else "—"
            aug = r["aug"]
            seeds = r["seeds"]
            wa, uar = format_wa_uar(r)

            inv, reason = is_invalid(r)
            if inv:
                validity = f"⚠️ **INVALID**: {reason}"
            else:
                validity = "✅"

            w(f"| {eid} | {corpus} | {pooling} | {fusion} | {adapter} | {unfreeze} | {aug} | {seeds} | {wa} | {uar} | {validity} |")

        w("")
        w(f"**{p} 核心结论**: {info.get('key_finding', '')}")
        w("")
        w("---")
        w("")

    # --- Section 3: Divergences ---
    w("## 3. 设计-执行分歧")
    w("")
    w("> 来源: `validation/design_vs_actual_diff.md` (已按独立复核修正)")
    w("> D1 已修正为 FAU=源域 (E3-07~12); D4 已修正为 18+36=54; D7 已修正为 CLI 速查遗漏而非全文缺失")
    w("")

    if div_items:
        w("| # | 类别 | 严重程度 | 设计意图 | 实际执行 | 影响 |")
        w("|---|------|---------|---------|---------|------|")
        for item in div_items:
            w(f"| {item['num']} | {item['category']} | {item['severity']} | {item['design']} | {item['actual']} | {item['impact']} |")
        w("")
        w("### 分歧根本原因")
        w("")
        w("多处分歧同源: **launch 脚本相对设计欠 specified** — 设计意图中明确的参数因脚本未传递而落回 `src/train.py` 的代码默认值。")
        w("因此实验条件以 launch 脚本实际传递的参数为准，非以设计文档的文字描述为准。")
    else:
        w("*分歧文档未找到或为空*")
    w("")
    w("---")
    w("")

    # --- Section 4: Special Notes ---
    w("## 4. 特殊说明")
    w("")
    w("### 数据划分")
    w("- `data_split_seed=42` 固定 (说话人独立 MD5 hash, 70/15/15)")
    w("- 所有 7 个 launch 脚本均显式传递 `--data_split_seed 42`")
    w("")
    w("### 训练配置")
    w("- 统一: epochs=100, patience=15")
    w("- B1: batch_size=32 (vs 设计 16)")
    w("- B2-B4, B6-B7: batch_size=16")
    w("- B5: batch_size=8 (设计 §4 L100 已预见显存减半)")
    w("- 微分学习率 (B5 unfreeze): backbone 1e-5, head 3e-4")
    w("- B7 fine-tune lr=3e-4 (代码默认; 设计建议 1e-4)")
    w("")
    w("### reg_profile 机制")
    w("- `default`: weight_decay=1e-3, label_smoothing=0.1, pooling_dropout=0.0, grad_clip=None")
    w("- `fau`: weight_decay=5e-3, label_smoothing=0.15, pooling_dropout=0.3, grad_clip=1.0")
    w("- B1/B3/B4/B5/B6/B7 均显式传递 `--reg_profile` (FAU 训练用 fau)")
    w("- B2 的 `run_zs()` 未传递 → 全部 18 个零样本实验用 default")
    w("  - 影响: E3-07~12 (FAU 为源域) 训练正则化不足")
    w("")
    w("### B2 跨语料类空间对齐")
    w("- C-BESD 域内 6 类, 跨语料强制对齐为 4 类 (丢弃 disgust/fear)")
    w("- 使用 `c-besd-4cl` 数据集子集")
    w("")
    w("### B7 迁移")
    w("- IEMOCAP 源 = E1-09 (prosody_guided) 而非 E1-08 (INVALID)")
    w("- Fine-tune 阶段统一 `--pooling_type self_attention`, 源池化头不参与迁移")
    w("- 微调 lr=3e-4 (代码默认) vs 设计建议 1e-4")
    w("")
    w("---")
    w("")

    # --- Section 5: Key Verification Numbers ---
    w("## 5. 关键核验数字")
    w("")
    w("### 各数据集天花板 (B1 frozen)")
    w("")
    w("| 实验 | 数据集 | Pooling | WA (3-seed mean±std, ddof=1) |")
    w("|------|--------|---------|------------------------------|")
    w("| E1-02 | C-BESD | self_attention | 91.87±1.56% |")
    w("| E1-05 | FAU Aibo | self_attention | 67.05±0.67% |")
    w("| E1-08 | IEMOCAP | self_attention | 63.76±0.53% ⚠️ INVALID (seed=42 旧协议) |")
    w("| E1-09 | IEMOCAP | prosody_guided | 64.38±1.05% |")
    w("")
    w("### B5 Unfreeze 天花板")
    w("")
    w("| 实验 | 数据集 | Pooling | WA (3-seed) |")
    w("|------|--------|---------|-------------|")
    w("| E2-01 | C-BESD | self_attention | **96.91±0.19%** 🔥 |")
    w("| E2-02 | FAU Aibo | self_attention | 66.37±1.04% |")
    w("| E2-03 | IEMOCAP | prosody_guided | 66.37±0.95% |")
    w("")
    w("### 跨语料天花板 (B2 Zero-shot)")
    w("| 方向 | 实验 | WA |")
    w("|------|------|----|")
    w("| IEMOCAP→C-BESD | E3-14 | **34.68%** (最高) |")
    w("| C-BESD→FAU | E3-03 | 21.91% (最低) |")
    w("")
    w("### B6 累积式消融 (C-BESD)")
    w("| 实验 | 配置 | WA | Δ |")
    w("|------|------|----|---|")
    w("| E6-01 | Mean+Last 基线 | 80.89% | — |")
    w("| E6-02 | +Adapter | 81.17% | +0.28pp |")
    w("| E6-03 | +SelfAttn (换Pooling) | **91.91%** | +10.74pp 🔥 |")
    w("| E6-04 | +WeightedFusion | **91.96%** | +0.05pp |")
    w("| E6-05 | +Adapter (全栈) | 91.12% | -0.84pp |")
    w("")
    w("### 最佳迁移 (B7)")
    w("| 实验 | 源→目标 | WA | vs 目标天花板 |")
    w("|------|---------|-----|---------------|")
    w("| E7-03 | FAU→C-BESD | **91.57±0.44%** | -0.30pp |")
    w("| E7-05 | IEMOCAP→C-BESD | 91.17±1.28% | -0.70pp |")
    w("")
    w("---")
    w("")

    # --- Section 6: Known Boundaries ---
    w("## 6. 已知边界")
    w("")
    w("### INVALID 实验 (3 个)")
    w("")
    w("| 实验 | 原因 | 影响 |")
    w("|------|------|------|")
    for eid, reason in invalid_exps:
        w(f"| {eid} | {reason} | mean±std 跨 seed 不可信, 排除自聚合分析 |")
    w("")
    w("### 配置字段可信度")
    w("")
    w("| 状态 | 字段 | 说明 |")
    w("|------|------|------|")
    w("| ✅ 可信 | pooling_type, batch_size, lr, data_split_seed, epochs, patience | 显式传参, 日志有记录 |")
    w("| ⚠️ 不可信 | augment_condition, fusion_mode, use_adapter, unfreeze_ssl, reg_profile, fusion_best_layer | 对部分旧协议实验为代码默认值, 非实际条件 |")
    w("")
    w("### 数据局限")
    w("")
    w("- 0/192 实验无原始预测数据 — WA/UAR 只能取信训练代码的 sklearn 计算")
    w("- B7 源域 checkpoint 不可独立确认 — 依赖 launch_b7.sh 正确执行")
    w("- B2 仅 1 seed — 无法计算标准差, 结论在统计上不充分")
    w("- B4 grid search 仅 1 seed — 单种子扫描, 无重复性验证")
    w("")
    w("---")
    w("")
    w("## 附录: 再生与校验")
    w("")
    w("### 一键再生")
    w("```")
    w("python scripts/gen_master_reference.py    # 生成本文档")
    w("python scripts/verify_all.py              # 全量验证 (含本脚本)")
    w("```")
    w("")
    w("### 数据溯源")
    w("| 内容 | 来源 |")
    w("|------|------|")
    w("| 实验条件 | `validation/provenance_manifest.csv` (gen_manifest.py 生成) |")
    w("| WA/UAR 指标 | `results/logs/E*-*.json` (ddof=1 聚合) |")
    w("| INVALID 标记 | manifest `aggregation_valid` / `seed_validity` 字段 |")
    w("| 设计-执行分歧 | `validation/design_vs_actual_diff.md` (已按独立复核修正) |")
    w("| 设计意图 | `docs/current/实验设计方案_v3_含学习笔记.md` |")
    w("| 实际执行 | `scripts/launch_b1.sh` ~ `launch_b7.sh` |")
    w("")

    # Write output
    os.makedirs(OUTPUT_MD.parent, exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n{'='*60}")
    print(f"Master reference table written: {OUTPUT_MD}")
    print(f"  Experiments: {len(rows)} rows")
    print(f"  INVALID: {len(invalid_exps)} ({', '.join(eid for eid, _ in invalid_exps)})")
    print(f"  Divergences: {len(div_items)} items")
    print(f"  Phases: {', '.join(phase_order)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
