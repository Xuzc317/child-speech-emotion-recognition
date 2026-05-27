"""Assemble paper_draft + publication_package into submission_bundle/ for review."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "submission_bundle"


def _copy_tree(src: Path, dst: Path) -> int:
    n = 0
    if not src.exists():
        return 0
    for p in src.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            n += 1
    return n


def main() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    sections = [
        "main.tex", "references.bib", "README.md",
        "0_Abstract.tex", "1_Introduction.tex", "2_Related_Work.tex",
        "3_Methodology.tex", "4_Experiments_and_Results.tex",
        "5_Analysis_and_Discussion.tex", "6_Conclusion.tex",
    ]
    draft = ROOT / "paper_draft"
    tex_dst = BUNDLE / "latex"
    tex_dst.mkdir()
    for name in sections:
        src = draft / name
        if src.exists():
            shutil.copy2(src, tex_dst / name)

    n_fig = _copy_tree(draft / "figures", BUNDLE / "figures")
    n_pub = _copy_tree(ROOT / "publication_package", BUNDLE / "publication_package")

    for doc in ("Full_Draft.docx", "Full_Draft_CN.docx"):
        src = draft / doc
        if src.exists():
            shutil.copy2(src, BUNDLE / doc)

    shutil.copy2(ROOT / "docs" / "json_verification_checklist.md", BUNDLE / "CHECKLIST.md")
    shutil.copy2(ROOT / "docs" / "autodl_local_inventory.md", BUNDLE / "AUTODL_INVENTORY.md")

    readme = BUNDLE / "README.txt"
    readme.write_text(
        "Submission bundle — auto-generated\n\n"
        "latex/          — compile main.tex from paper_draft (copy main.tex here)\n"
        "figures/        — all paper figures\n"
        "publication_package/ — JSON, CSV, XAI npz\n"
        "Full_Draft*.docx — Word drafts for team review\n"
        "CHECKLIST.md    — experiment JSON verification\n",
        encoding="utf-8",
    )
    shutil.copy2(draft / "main.tex", tex_dst / "main.tex")

    print(f"Bundle written to {BUNDLE}")
    print(f"  tex files: {len(list(tex_dst.glob('*.tex')))}")
    print(f"  figures: {n_fig}")
    print(f"  publication_package files: {n_pub}")


if __name__ == "__main__":
    main()
