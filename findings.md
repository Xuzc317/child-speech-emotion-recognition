# Findings — Session 2026-05-18

## Codebase-Stand (Initialanalyse)

### Architektur (bestätigt)
- Frozen WavLM Base+ → `WavLMLayerFusion` (12 Layer) → Pooling (111,105 Params) → `SEMLP`
- Training: `src/train.py` (online WavLM, nicht legacy `train_ssl.py`)

### Regularisierung (Lücke)
| Komponente | Aktuell |
|------------|---------|
| `TemporalImportancePooling` / `SelfAttentionPooling` | **Kein Dropout** |
| `SEMLP` | Dropout 0.3 / 0.3 / 0.2 |
| `train.py` | `weight_decay=1e-3`, `label_smoothing=0.1`, `patience=15` |

→ Erklärt schnelles Overfitting auf FAU (best_epoch=2 für Exp5/5b).

### Git-Status (überraschend)
- HEAD: `0d5de86` — docs summary bereits committed
- `results/logs/exp{1..5b}*.json` **bereits getrackt**
- `src/train.py`: **kein Diff zu HEAD** (Initial-Snapshot `M train.py` veraltet oder bereits committed)
- Untracked: `checkpoints/`, `fau_aibo.tar.gz`, `results_remote/`

### Datensätze im Code
- `fau-aibo` in `dataset.py` mit lokalem Pfad `D:\大学\数据集IS2009EmotionChallenge\...`
- Kein `prepare_extra_datasets.py` vorhanden

### Publication / Paper
- `publication_package/` vorhanden; **fehlt** `exp5b_self_attention_fau.json` in logs/
- `paper_draft/`: nur `Methods_and_Results_Skeleton.tex`; Ziel-Dateien 4 & 5 noch nicht erstellt

## Experimentmatrix (Nutzer, bestätigt in `docs/全部实验结果汇总.md`)

| Exp | Train | Test | Pooling | Test WA | best_epoch |
|-----|-------|------|---------|---------|------------|
| 1 | C-BESD | C-BESD | Self-Attn | 79.63% | — |
| 2 | C-BESD | C-BESD | Prosody | 92.04% | — |
| 3 | IEMOCAP | IEMOCAP | Prosody | 60.14% | — |
| 4 | C-BESD | FAU | Prosody | **16.68%** | 16 |
| 5 | FAU | FAU | Prosody | 68.87% | **2** |
| 5b | FAU | FAU | Self-Attn | 69.54% | **2** |

## Wissenschaftliche Narrative (für Paper §4–5)
1. **Acted vs. spontaneous**: 92% vs ~69% — Distribution shift, nicht nur „schwierigeres Labeling“
2. **Zero-shot collapse (16.68%)**: Priors aus acted Malay child speech transferieren nicht zu spontaneous German child speech
3. **Prosody auf FAU**: leicht schlechter als Self-Attn → Hypothese: Prosody-Netz lernt akustisches Rauschen / Sprecher-Leakage ohne stärkere Reg
4. **FD-Accuracy**: konsistent mit Phase 5 (C-BESD vs FAU FD ~12–16)

## Externe Datensätze (vorläufig)
| Dataset | Zugang | Emotion-Labels für SER |
|---------|--------|------------------------|
| MyST (My Science Tutor) | LDC / Edu license | Eher ASR/Tutoring; Emotion begrenzt — eher Domain-Diversity |
| EmoReact | Academic request | Video+Audio, Kinder — Emotion-relevant |
| KidsTALC | Zu verifizieren | Kindersprache, ggf. kein 4-Klassen-SER |

→ Skript wird versuchen Open-Mirror/HF; sonst `docs/dataset_instructions.md`.
