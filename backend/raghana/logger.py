"""Structured per-stage logging to JSONL + stdout."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

_ARTIFACTS = Path(__file__).parent.parent / "artifacts"
_ARTIFACTS.mkdir(exist_ok=True)
_JSONL_PATH = _ARTIFACTS / "pipeline.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] raghana: %(message)s",
)
_logger = logging.getLogger("raghana")


def log_stage(
    stage: str,
    duration_ms: float,
    input_preview: str,
    output_preview: str,
) -> None:
    """Append one JSONL record and echo to stdout."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "duration_ms": round(duration_ms, 2),
        "input_preview": str(input_preview)[:200],
        "output_preview": str(output_preview)[:200],
    }
    with open(_JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    _logger.info("[%s] %.0fms | %s", stage, duration_ms, str(output_preview)[:80])
