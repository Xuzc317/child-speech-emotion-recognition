# Extra Naturalistic Child-Speech Datasets (MyST / KidsTALC / EmoReact)

This project can probe extra naturalistic child speech corpora using:
- **MyST (My Science Tutor)**
- **KidsTALC**
- **EmoReact**

Because these datasets may require license approval or authenticated access, use the instructions below.

## 1) MyST (My Science Tutor)

- Primary overview: [CMU MyST project](https://www.cs.cmu.edu/~ILSLAB/projects/MyST/)
- Typical distribution channel: LDC / institutional request (varies by release)
- If your institution has access, download to:
  - `data/external_datasets/myst/`

Suggested structure:
- `data/external_datasets/myst/audio/`
- `data/external_datasets/myst/metadata/`

## 2) KidsTALC

- Reference page: [KidsTALC resources](https://www.cstr.ed.ac.uk/projects/kidstalc/)
- Access is commonly via academic request or controlled release.
- Download and extract to:
  - `data/external_datasets/kidstalc/`

Suggested structure:
- `data/external_datasets/kidstalc/wav/`
- `data/external_datasets/kidstalc/labels.csv`

## 3) EmoReact

- Project/reference: [EmoReact dataset page](https://www2.informatik.uni-hamburg.de/wtm/EmoReact/)
- Access generally requires contacting dataset maintainers.
- Download and extract to:
  - `data/external_datasets/emoreact/`

Suggested structure:
- `data/external_datasets/emoreact/audio/`
- `data/external_datasets/emoreact/annotations/`

## 4) Auto-probe helper script

Run:

```bash
python -m src.data.prepare_extra_datasets --output_dir data/external_datasets
```

This script will:
- try lightweight URL probes where available,
- write `data/external_datasets/download_manifest.json`,
- and indicate which corpora still require manual download.

## 5) Notes for experiments

- Use strong regularization profile (`--reg_profile fau`) for spontaneous/naturalistic corpora.
- Keep default profile for acted C-BESD experiments (Exp1/Exp2) to preserve baseline comparability.
