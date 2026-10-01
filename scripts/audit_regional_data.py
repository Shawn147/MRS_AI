"""Validate downloaded regional research cohorts, provenance and prepared counts."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.regional_data import REGIONAL_DIR, audit_regional


def main():
    report = audit_regional()
    (REGIONAL_DIR / 'quality_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
