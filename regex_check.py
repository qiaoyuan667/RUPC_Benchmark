#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Tuple


# =========================================================
# Normalization
# =========================================================

def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def normalize_text(text: str) -> str:
    text = safe_str(text).lower().strip()

    # normalize common punctuation/spaces
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("/", " / ")
    text = text.replace("-", " - ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def alnum_only(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


# =========================================================
# Recursive extraction of leaf values
# =========================================================

def flatten_leaf_values(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    """
    Recursively extract leaf scalar values from nested dict/list structures.
    Returns list of (path, value_as_string).
    """
    out: List[Tuple[str, str]] = []

    if obj is None:
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            out.extend(flatten_leaf_values(v, child_prefix))
        return out

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            child_prefix = f"{prefix}[{i}]"
            out.extend(flatten_leaf_values(v, child_prefix))
        return out

    s = safe_str(obj).strip()
    if s:
        out.append((prefix, s))
    return out


def should_check_value(value: str) -> bool:
    v = normalize_text(value)
    if not v:
        return False
    if len(alnum_only(v)) < 3:
        return False
    return True


# =========================================================
# Regex generation
# =========================================================

def value_to_flexible_pattern(value: str) -> str:
    """
    Build a forgiving regex for a structured value.
    Useful for names, phones, dates, ids, money, etc.
    """
    v = safe_str(value).strip()
    if not v:
        return ""

    p = re.escape(v.lower())

    # allow flexible whitespace
    p = p.replace(r"\ ", r"\s+")

    # common separators become flexible
    p = p.replace(r"\-", r"[\s\-]*")
    p = p.replace(r"\/", r"[\s\/\-]*")
    p = p.replace(r"\.", r"[\.\s]*")
    p = p.replace(r"\,", r"[\s,]*")
    p = p.replace(r"\:", r"[\s:]*")

    return p


def contains_value(text: str, value: str) -> bool:
    """
    Multi-strategy containment check.
    """
    if not value or not text:
        return False

    norm_text = normalize_text(text)
    norm_value = normalize_text(value)

    # Strategy 1: normalized substring
    if norm_value in norm_text:
        return True

    # Strategy 2: alnum-only containment
    av = alnum_only(value)
    at = alnum_only(text)
    if av and len(av) >= 4 and av in at:
        return True

    # Strategy 3: flexible regex
    pattern = value_to_flexible_pattern(value)
    if pattern and re.search(pattern, norm_text, flags=re.IGNORECASE):
        return True

    return False


# =========================================================
# Validation for one field
# =========================================================

def validate_values_in_text(
    text: str,
    values_obj: Any,
    label: str
) -> Dict[str, Any]:
    leaf_values = flatten_leaf_values(values_obj)
    checked = []
    missing = []

    for path, value in leaf_values:
        if not should_check_value(value):
            continue

        ok = contains_value(text, value)
        item = {
            "path": path,
            "value": value,
            "found": ok,
        }
        checked.append(item)
        if not ok:
            missing.append(item)

    total = len(checked)
    found_count = sum(1 for x in checked if x["found"])
    coverage = (found_count / total) if total else 1.0

    return {
        "label": label,
        "total": total,
        "found": found_count,
        "coverage": coverage,
        "missing": missing,
        "checked": checked,
        "is_complete": len(missing) == 0,
    }


# =========================================================
# Sample-level validation
# =========================================================

def get_nested(d: Dict[str, Any], *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def validate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    source_document_text = get_nested(sample, "generated_texts", "source_document_text", default="") or ""

    do_not_disclose_values = sample.get("do_not_disclose_values", {})
    allowed_values = sample.get("allowed_values", {})

    dnd_result = validate_values_in_text(
        source_document_text,
        do_not_disclose_values,
        "do_not_disclose_values"
    )

    allowed_result = validate_values_in_text(
        source_document_text,
        allowed_values,
        "allowed_values"
    )

    return {
        "sample_id": sample.get("sample_id"),
        "source_document_text_exists": bool(source_document_text.strip()),
        "source_document_text_valid": (
            dnd_result["is_complete"] and allowed_result["is_complete"]
        ),
        "do_not_disclose_values_check": dnd_result,
        "allowed_values_check": allowed_result,
    }


# =========================================================
# File-level validation
# =========================================================

def validate_dataset(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array")

    results = []
    invalid_count = 0

    for sample in data:
        result = validate_sample(sample)
        results.append(result)
        if not result["source_document_text_valid"]:
            invalid_count += 1

    summary = {
        "total_samples": len(results),
        "invalid_samples": invalid_count,
        "valid_samples": len(results) - invalid_count,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved validation report to: {output_path}")
    print(f"Total samples: {len(results)}")
    print(f"Invalid samples: {invalid_count}")
    print(f"Valid samples: {len(results) - invalid_count}")


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Regex-based validation for generated_texts.source_document_text"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file, e.g. .\\data\\privacy_benchmark_rendered.json"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output report JSON file, e.g. .\\data\\privacy_benchmark_validation_report.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_dataset(
        input_path=args.input,
        output_path=args.output
    )