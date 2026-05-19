# Aufgabenplan: INTERSPEECH/ICASSP 2026 — Distribution-Driven Child SER (Finalisierung)

## Ziel
Repository aufräumen, Regularisierung gegen FAU-Epoch-2-Overfitting verschärfen, optionale Kinder-Sprachdatensätze vorbereiten, AutoDL-Experimente (Exp5/5b + ggf. neue Korpora) wiederholen, Ergebnisse nach `publication_package/` synchronisieren und ehrliche LaTeX-Abschnitte für Experimente & Diskussion verfassen.

## Aktuelle Phase
Phase 1 — Planung & Nutzerfreigabe (keine Ausführung bis Bestätigung)

## Phasen

### Phase 1: Repository & Git-Hygiene
- [x] Codebase & Git-Status analysiert
- [ ] `.gitignore` erweitern: `checkpoints/`, `fau_aibo.tar.gz`, `results_remote/`
- [ ] Commit: nur wenn tatsächliche Änderungen (Hinweis: `src/train.py` und `results/logs/*.json` sind bereits in HEAD `0d5de86`)
- **Status:** pending (wartet auf Freigabe)

### Phase 2: Regularisierung & Datensatz-Vorbereitung
- [ ] `src/train.py`: CLI für stärkere Regularisierung (`weight_decay`, `label_smoothing`, `grad_clip`, ggf. reduzierte `patience` für kleine Korpora)
- [ ] `src/models/pooling.py`: Dropout in Prosody- und Self-Attention-Pfaden (Param-Anzahl 111,105 unverändert)
- [ ] `src/data/prepare_extra_datasets.py` + `docs/dataset_instructions.md` (MyST / KidsTALC / EmoReact)
- [ ] `dataset.py` / `label_mapper.py`: Registry-Einträge für erfolgreich geladene Korpora
- **Status:** pending

### Phase 3: AutoDL — Exp5 / Exp5b (stark regularisiert)
- [ ] Code + Skripte hochladen (rsync/scp)
- [ ] FAU Aibo Pfad auf Instanz verifizieren
- [ ] Exp5 (prosody) & Exp5b (self_attention) mit neuen Hyperparametern
- [ ] Ggf. Ablation auf neuen Kinder-Korpora
- **Status:** pending — **blockiert ohne echte SSH-Zugangsdaten**

### Phase 4: Sync & LaTeX
- [ ] JSON/CSV → `publication_package/`
- [ ] `paper_draft/4_Experiments_and_Results.tex`
- [ ] `paper_draft/5_Analysis_and_Discussion.tex` (16.68% Zero-Shot, Epoch-2-Overfitting als Hauptbefunde)
- **Status:** pending

## Schlüsselfragen
1. AutoDL: echte SSH-Befehlszeile + bevorzugt SSH-Key statt Passwort im Chat?
2. Sollen Regularisierungs-Defaults global verschärft werden oder nur per `--reg_profile fau`?
3. Welche zusätzlichen Datensätze liegen lokal bereits vor (Pfade)?

## Getroffene Entscheidungen
| Entscheidung | Begründung |
|-------------|-----------|
| Dropout in Pooling, nicht mehr MLP-Layer | Overfitting tritt bei Prosody-Pfad auf; Param-Parität 111,105 bleibt mit Dropout erhalten |
| `weight_decay` 1e-3 → 5e-3 (FAU-Profil) | Kleines spontanes Korpus, schneller Val-Peak bei Epoch 2 |
| Manuelle Download-Docs für Lizenzierte Korpora | MyST/EmoReact erfordern oft Registrierung |
| Keine Passwörter in Repo/Scripts | Sicherheit |

## Aufgetretene Fehler
| Fehler | Versuch | Lösung |
|--------|---------|--------|
| — | — | — |

## Hinweise
- Nach Freigabe: Phase 1 → 2 → 3 → 4 sequenziell
- Vor AutoDL: lokaler Smoke-Test (1 Epoch, kleines Subset)
