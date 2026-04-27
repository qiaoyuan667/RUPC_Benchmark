#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# =========================================================
# SwissAI client
# =========================================================

_client: Optional[OpenAI] = None


def build_client() -> OpenAI:
    api_key = os.getenv("SWISS_AI_API_KEY")
    base_url = os.getenv("SWISS_AI_BASE_URL")

    if not api_key:
        raise EnvironmentError("Missing SWISS_AI_API_KEY environment variable.")

    if not base_url:
        raise EnvironmentError("Missing SWISS_AI_BASE_URL environment variable.")

    return OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout=120,
        max_retries=1,
    )


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = build_client()
    return _client


# =========================================================
# I/O
# =========================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


# =========================================================
# Missing value extraction
# =========================================================

def extract_missing_source_values(validation_result: Dict[str, Any]) -> List[str]:
    field_results = validation_result.get("field_results", {}) or {}
    source_result = field_results.get("source_document_text", {}) or {}
    checks = source_result.get("checks", {}) or {}

    missing_values: List[str] = []

    dnd_check = checks.get("do_not_disclose_values_present", {}) or {}
    allowed_check = checks.get("allowed_values_present", {}) or {}

    for item in dnd_check.get("missing", []) or []:
        value = safe_str(item.get("value"))
        if value:
            missing_values.append(value)

    for item in allowed_check.get("missing", []) or []:
        value = safe_str(item.get("value"))
        if value:
            missing_values.append(value)

    seen = set()
    deduped = []
    for v in missing_values:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(v)

    return deduped


def build_invalid_map(validation_report: Dict[str, Any]) -> Dict[str, List[str]]:
    invalid_map: Dict[str, List[str]] = {}

    for result in validation_report.get("results", []) or []:
        sample_id = result.get("sample_id")
        if not sample_id:
            continue

        field_results = result.get("field_results", {}) or {}
        source_result = field_results.get("source_document_text", {}) or {}

        if source_result.get("is_valid"):
            continue

        missing_values = extract_missing_source_values(result)
        if missing_values:
            invalid_map[sample_id] = missing_values

    return invalid_map


# =========================================================
# LLM repair
# =========================================================

def build_source_repair_prompt(
    original_source: str,
    missing_values: List[str],
    sample: Dict[str, Any],
) -> str:
    domain = sample.get("domain", "unknown")
    metadata = sample.get("metadata", {}) or {}
    source_inputs = sample.get("source_document_inputs", {}) or {}

    return f"""
You are repairing a generated source document for a privacy benchmark.

Your task:
Revise the source document so that it naturally includes every required missing value.

Rules:
- Preserve the original document's domain, style, tone, and structure.
- Keep it realistic and about one page.
- Do not turn it into a privacy policy.
- Do not add disclosure rules, redaction instructions, or privacy warnings.
- Do not explain the repair.
- You MUST include every missing value exactly as written.
- Do not omit, paraphrase away, generalize, or replace the missing values.
- The repaired document should still read like a realistic source document.
- Output only the repaired source document.

Domain:
{domain}

Metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Source document inputs:
{json.dumps(source_inputs, ensure_ascii=False, indent=2)}

Original source document:
{original_source}

Missing values that must be included exactly:
{json.dumps(missing_values, ensure_ascii=False, indent=2)}
""".strip()


def call_llm(
    prompt: str,
    model: str,
    temperature: float = 0.6,
    max_tokens: int = 3200,
    retries: int = 2,
    sleep_seconds: float = 1.0,
) -> str:
    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            resp = get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = resp.choices[0].message.content
            if content and content.strip():
                return content.strip()

            raise RuntimeError("Empty model output.")

        except Exception as e:
            last_error = e
            print(f"[repair attempt {attempt + 1}/{retries + 1}] {type(e).__name__}: {e}")

            if attempt < retries:
                time.sleep(sleep_seconds * (attempt + 1))

    raise RuntimeError(f"LLM repair failed after retries: {last_error}")


def repair_source_document(
    sample: Dict[str, Any],
    missing_values: List[str],
    model: str,
) -> str:
    gt = sample.get("generated_texts", {}) or {}
    original_source = safe_str(gt.get("source_document_text", ""))

    prompt = build_source_repair_prompt(
        original_source=original_source,
        missing_values=missing_values,
        sample=sample,
    )

    return call_llm(
        prompt=prompt,
        model=model,
        temperature=0.6,
        max_tokens=3200,
    )


# =========================================================
# Repair dataset
# =========================================================

def repair_dataset(
    rendered_path: str,
    validation_report_path: str,
    output_path: str,
    model: str,
    max_workers: int = 5,
) -> None:
    rendered_data = load_json(rendered_path)
    validation_report = load_json(validation_report_path)

    if not isinstance(rendered_data, list):
        raise ValueError("Rendered dataset must be a JSON array.")

    invalid_map = build_invalid_map(validation_report)

    print(f"Loaded rendered dataset: {len(rendered_data)} samples")
    print(f"Samples needing source_document repair: {len(invalid_map)}")
    print(f"Repair model: {model}")
    print(f"Max workers: {max_workers}")

    results = rendered_data[:]

    def repair_one(idx: int, sample: Dict[str, Any]) -> Tuple[int, Dict[str, Any], bool, Optional[str]]:
        sample_id = sample.get("sample_id")

        if sample_id not in invalid_map:
            return idx, sample, False, None

        missing_values = invalid_map[sample_id]

        try:
            repaired_source = repair_source_document(
                sample=sample,
                missing_values=missing_values,
                model=model,
            )

            sample.setdefault("generated_texts", {})
            sample["generated_texts"]["source_document_text"] = repaired_source

            sample.setdefault("generation_meta", {})
            sample["generation_meta"]["source_document_repaired"] = True
            sample["generation_meta"]["source_document_repair_model"] = model
            sample["generation_meta"]["source_document_repair_missing_values"] = missing_values

            return idx, sample, True, None

        except Exception as e:
            sample.setdefault("generation_meta", {})
            sample["generation_meta"]["source_document_repaired"] = False
            sample["generation_meta"]["source_document_repair_error"] = str(e)
            sample["generation_meta"]["source_document_repair_missing_values"] = missing_values

            return idx, sample, False, str(e)

    repaired_count = 0
    skipped_count = 0
    failed_count = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(repair_one, idx, sample): (idx, sample)
            for idx, sample in enumerate(rendered_data)
        }

        for future in as_completed(futures):
            idx, sample = futures[future]
            completed += 1
            sample_id = sample.get("sample_id")

            try:
                result_idx, repaired_sample, repaired, error = future.result()
                results[result_idx] = repaired_sample
            except Exception as e:
                repaired = False
                error = str(e)
                failed_count += 1
                print(f"[{completed}/{len(rendered_data)}] ERROR sample_id={sample_id}: {error}")
                continue

            if sample_id not in invalid_map:
                skipped_count += 1
                continue

            if repaired:
                repaired_count += 1
                print(
                    f"[{completed}/{len(rendered_data)}] REPAIRED "
                    f"sample_id={sample_id} | missing={len(invalid_map[sample_id])}"
                )
            else:
                failed_count += 1
                print(
                    f"[{completed}/{len(rendered_data)}] FAILED "
                    f"sample_id={sample_id} | error={error}"
                )

    save_json(results, output_path)

    print("\nRepair finished.")
    print(f"Repaired samples: {repaired_count}")
    print(f"Skipped samples: {skipped_count}")
    print(f"Failed repairs: {failed_count}")
    print(f"Saved repaired dataset to: {output_path}")


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Repair invalid source_document_text fields using SwissAI LLM."
    )

    parser.add_argument(
        "--rendered",
        required=True,
        help="Path to rendered dataset JSON, e.g. .\\data\\privacy_benchmark_rendered.json",
    )

    parser.add_argument(
        "--validation-report",
        required=True,
        help="Path to validation report JSON, e.g. .\\data\\privacy_benchmark_validation_report.json",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output repaired rendered dataset JSON.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of source documents to repair in parallel.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="SwissAI model used for source document repair.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    repair_dataset(
        rendered_path=args.rendered,
        validation_report_path=args.validation_report,
        output_path=args.output,
        model=args.model,
        max_workers=args.max_workers,
    )