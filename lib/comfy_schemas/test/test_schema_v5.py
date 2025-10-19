import json
from pathlib import Path
from typing import Any, Dict

import pytest

from lib.metadata import read_comfy_metadata
from lib.comfy_schemas.comfy_analysis_v2 import extract_from_prompt


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "lib" / "comfy_schemas" / "schemas" / "schema_v5.yml"
EXPORT_PATH = ROOT / "lib" / "comfy_schemas" / "test" / "data" / "test_schema_v5.png"


def _run_validation() -> tuple[str, str]:
    """Run validation; return (status, message). status in {ok, extract-failed, values-failed}."""
    expected = {
        "model": "sd_xl_base_1.0.safetensors",
        "loras": "[loras]",
        "positive": "[positive]",
        "negative": "[negative]",
        "IPAdapter": "disabled",
        "ip_image": "strolling_a.jpg",
        "ip_weight": 0.6,
        "dim": "896x1152",
        "steps": 30,
        "cfg": 5.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "rescale_cfg": False,
        "perp_neg": False,
    }

    prompt, _ = read_comfy_metadata(EXPORT_PATH)
    # print("PROMPT", prompt)

    try:
        extracted = extract_from_prompt(prompt, SCHEMA_PATH)
    except Exception as e:
        msg = (
            "Extraction failed with schema_v5.\n"
            f"  Error: {e.__class__.__name__}: {e}\n"
            "  Hint: Ensure schema_v5.yml has correct node ids/types and input names for this export."
        )
        return ("extract-failed", msg)

    mapping = {
        "checkpoint": "model",
        "loras": "loras",
        "positive_prompt": "positive",
        "negative_prompt": "negative",
        "steps": "steps",
        "cfg": "cfg",
        "sampler_name": "sampler_name",
        "scheduler": "scheduler",
        "aspect_ratio": None,
        "swap_dimensions": None,
        "rescale_cfg": "rescale_cfg",
        "perp_neg": "perp_neg",
        "ip_enabled": "IPAdapter",
        "ip_image": "ip_image",
        "ip_weight": "ip_weight",
    }

    derived: Dict[str, Any] = {}
    ar_label = extracted.get("aspect_ratio")
    if isinstance(ar_label, str) and "896x1152" in ar_label:
        derived["dim"] = "896x1152"
    if "ip_enabled" in extracted:
        derived["IPAdapter"] = "enabled" if extracted.get("ip_enabled") else "disabled"

    comparable: Dict[str, Any] = {}
    for src_key, exp_key in mapping.items():
        if exp_key is None:
            continue
        if src_key in extracted:
            comparable[exp_key] = extracted[src_key]
    comparable.update(derived)

    missing = []
    mismatches = []
    for key, exp_val in expected.items():
        if key not in comparable:
            missing.append(key)
        else:
            got = comparable[key]
            if got != exp_val:
                mismatches.append((key, exp_val, got))

    if missing or mismatches:
        lines = ["Schema v5 validation results:"]
        if missing:
            lines.append("  Missing fields:")
            lines.extend([f"    - {k}" for k in missing])
        if mismatches:
            lines.append("  Mismatched values:")
            for k, exp, got in mismatches:
                got_str = str(got)
                if len(got_str) > 100:
                    got_str = got_str[:100] + "…"
                lines.append(f"    - {k}: expected={exp!r}, got={got_str!r}")
        lines.append("  Note: Update schema_v5.yml role ids/types and outputs to address the above.")
        return ("values-failed", "\n".join(lines))

    return ("ok", "Schema v5 validation PASSED.")


def test_schema_v5():
    status, msg = _run_validation()
    if status != "ok":
        pytest.fail(msg)
    # If ok, test passes silently
