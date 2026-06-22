#!/usr/bin/env python3
"""
One-command full verification pipeline.
Rebuilds all validation artifacts from raw experiment logs.
Equivalent to: make verify
"""

import subprocess, sys
from pathlib import Path

SCRIPTS = [
    ("gen_manifest.py", "Rebuild provenance manifest"),
    ("regen_handbook.py", "Regenerate authoritative handbook"),
    ("check_metrics.py", "Check metric completeness"),
    ("phase4_audit.py", "Run reproducibility audit"),
    ("gen_master_reference.py", "Generate master reference table"),
]

def main():
    script_dir = Path(__file__).resolve().parent
    passed = 0
    failed = 0

    for script, desc in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"  {script}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, str(script_dir / script)],
            capture_output=True, text=True,
            cwd=str(script_dir.parent)
        )
        if result.returncode == 0:
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print(f"  [PASS] {script}")
            passed += 1
        else:
            print(result.stderr[-500:])
            print(f"  [FAIL] {script} (exit {result.returncode})")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  VERIFICATION COMPLETE")
    print(f"  Passed: {passed}/{len(SCRIPTS)}")
    print(f"  Failed: {failed}/{len(SCRIPTS)}")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
