"""Create an auditable promotion-candidate record when Model Registry is unavailable."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    return parser.parse_args()


def find_evaluation(path: str | Path) -> Path:
    source = Path(path)
    if source.is_file():
        return source
    matches = sorted(source.rglob("evaluation.json"))
    if not matches:
        raise FileNotFoundError(f"evaluation.json not found below {source}")
    return matches[0]


def main() -> None:
    args = parse_args()
    evaluation = json.loads(find_evaluation(args.evaluation).read_text(encoding="utf-8"))
    if int(evaluation.get("gate", {}).get("passed", 0)) != 1:
        raise ValueError("Cannot record a promotion candidate when the quality gate failed")
    record = {
        "schema_version": "aai540-promotion-candidate-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approval_status": "PendingManualApproval",
        "evaluation": evaluation,
        "note": "S3 fallback record; review before any deployment or champion change.",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "promotion_candidate.json").write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
