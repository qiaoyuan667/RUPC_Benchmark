#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object")
    return data


def avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"none", "null", "nan", ""}:
        return None
    try:
        return float(x)
    except Exception:
        return None


def compute_example_score(
    privacy_score: float | None,
    utility_score: float | None,
) -> float | None:
    """
    Rules:
    - both None -> None, invalid example
    - only privacy exists -> privacy
    - only utility exists -> utility
    - both exist -> average of privacy and utility
    """
    if privacy_score is None and utility_score is None:
        return None
    if privacy_score is None:
        return utility_score
    if utility_score is None:
        return privacy_score
    return (privacy_score + utility_score) / 2.0


def summarize_examples(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    privacy_scores: List[float] = []
    utility_scores: List[float] = []
    example_scores: List[float] = []

    leaked_counts = []
    allowed_counts = []
    covered_dnd_counts = []
    covered_allowed_counts = []

    invalid_example_count = 0

    for ex in examples:
        p = safe_float(ex.get("privacy_score"))
        u = safe_float(ex.get("utility_score"))

        # Do NOT trust stored example_score here.
        # Recompute it so both-None examples become None and are excluded.
        e = compute_example_score(p, u)

        if p is not None:
            privacy_scores.append(p)
        if u is not None:
            utility_scores.append(u)

        if e is not None:
            example_scores.append(e)
        else:
            invalid_example_count += 1

        leaked_counts.append(int(ex.get("leaked_count", 0) or 0))
        allowed_counts.append(int(ex.get("allowed_count", 0) or 0))
        covered_dnd_counts.append(int(ex.get("covered_do_not_disclose_count", 0) or 0))
        covered_allowed_counts.append(int(ex.get("covered_allowed_count", 0) or 0))

    valid_example_count = len(examples) - invalid_example_count

    return {
        "num_examples": len(examples),
        "valid_example_count": valid_example_count,

        "avg_privacy_score": avg(privacy_scores),
        "privacy_score_100": 100.0 * avg(privacy_scores),

        "avg_utility_score": avg(utility_scores),
        "utility_score_100": 100.0 * avg(utility_scores),

        "avg_example_score": avg(example_scores),
        "example_score_100": 100.0 * avg(example_scores),

        "total_leaked_count": sum(leaked_counts),
        "total_allowed_count": sum(allowed_counts),
        "total_covered_do_not_disclose_count": sum(covered_dnd_counts),
        "total_covered_allowed_count": sum(covered_allowed_counts),
    }


def add_group(
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]],
    scope: str,
    domain: str,
    privacy_level: str,
    attack_strength: str,
    ex: Dict[str, Any],
) -> None:
    key = (scope, domain, privacy_level, attack_strength)
    groups[key].append(ex)


def build_divided_summary(details: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "models": {}
    }

    models = details.get("models", {}) or {}

    for model_name, model_block in models.items():
        examples = model_block.get("examples", []) or []
        groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)

        for ex in examples:
            domain = str(ex.get("domain", "unknown"))
            privacy_level = str(ex.get("privacy_level", "unknown"))
            attack_strength = str(ex.get("attack_strength", "unknown"))

            add_group(groups, "overall", "", "", "", ex)
            add_group(groups, "by_domain", domain, "", "", ex)
            add_group(groups, "by_privacy_level", "", privacy_level, "", ex)
            add_group(groups, "by_attack_strength", "", "", attack_strength, ex)
            add_group(groups, "by_domain_privacy_level", domain, privacy_level, "", ex)
            add_group(groups, "by_domain_attack_strength", domain, "", attack_strength, ex)
            add_group(groups, "by_privacy_attack_strength", "", privacy_level, attack_strength, ex)
            add_group(groups, "by_domain_privacy_attack_strength", domain, privacy_level, attack_strength, ex)

        rows = []
        for (scope, domain, privacy_level, attack_strength), group_examples in sorted(groups.items()):
            summary = summarize_examples(group_examples)
            rows.append({
                "model_name": model_name,
                "scope": scope,
                "domain": domain,
                "privacy_level": privacy_level,
                "attack_strength": attack_strength,
                **summary,
            })

        output["models"][model_name] = {
            "num_examples": len(examples),
            "divided_score_summary": rows,
        }

    return output


def flatten_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for _, model_block in (summary.get("models", {}) or {}).items():
        for row in model_block.get("divided_score_summary", []) or []:
            rows.append(row)
    return rows


def save_json(data: Any, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_csv_fieldnames(include_source_file: bool = False) -> List[str]:
    fieldnames = [
        "model_name",
        "scope",
        "domain",
        "privacy_level",
        "attack_strength",
        "num_examples",
        "valid_example_count",
        "avg_privacy_score",
        "privacy_score_100",
        "avg_utility_score",
        "utility_score_100",
        "avg_example_score",
        "example_score_100",
        "total_leaked_count",
        "total_allowed_count",
        "total_covered_do_not_disclose_count",
        "total_covered_allowed_count",
    ]

    if include_source_file:
        return ["source_file", *fieldnames]

    return fieldnames


def save_csv(rows: List[Dict[str, Any]], path: str, include_source_file: bool = False) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = get_csv_fieldnames(include_source_file=include_source_file)

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build divided privacy/utility summaries for all details*.json files."
    )
    parser.add_argument("--input-dir", default="data/leaderboard")
    parser.add_argument("--output-dir", default="data/leaderboard_divided_summary")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_files = sorted(input_dir.glob("details*.json"))

    if not detail_files:
        print(f"No details*.json files found in {input_dir}")
        raise SystemExit(0)

    all_rows = []

    for details_path in detail_files:
        try:
            details = load_json(str(details_path))
            summary = build_divided_summary(details)
            rows = flatten_summary(summary)

            stem = details_path.stem
            output_json = output_dir / f"{stem}_divided_summary.json"
            output_csv = output_dir / f"{stem}_divided_summary.csv"

            save_json(summary, str(output_json))
            save_csv(rows, str(output_csv))

            all_rows.extend([
                {"source_file": details_path.name, **row}
                for row in rows
            ])

            print(f"Converted {details_path.name}: rows={len(rows)}")

        except Exception as e:
            print(f"Failed {details_path.name}: {e}")

    if all_rows:
        combined_csv = output_dir / "all_details_divided_summary.csv"
        save_csv(all_rows, str(combined_csv), include_source_file=True)
        print(f"Saved combined CSV to: {combined_csv}")