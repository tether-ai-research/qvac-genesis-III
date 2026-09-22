"""Postprocessors for the Genesis III LLM-as-a-parser evaluation."""

import re

from opencompass.registry import DICT_POSTPROCESSORS, TEXT_POSTPROCESSORS


THINKING_END_MARKERS = [
    "</think>",
    "<unused95>",
    "</reasoning>",
    "</thought>",
    "[/THINK]",
]
THINKING_START_MARKERS = [
    "<think>",
    "<unused94>",
    "<reasoning>",
    "<thought>",
    "[THINK]",
]
MAX_PREDICTION_CHARS = 20_000


def strip_thinking_content(text):
    if not text:
        return text

    found_end_marker = False
    for marker in THINKING_END_MARKERS:
        index = text.lower().rfind(marker.lower())
        if index != -1:
            found_end_marker = True
            after_marker = text[index + len(marker) :].strip()
            if after_marker:
                text = after_marker
                break

    if not found_end_marker:
        lowered = text.lower()
        if any(
            lowered.startswith(marker.lower())
            for marker in THINKING_START_MARKERS
        ):
            text = "NO_ANSWER"

    if len(text) > MAX_PREDICTION_CHARS:
        text = text[:MAX_PREDICTION_CHARS]
        text += "\n[... output truncated for judge ...]"
    return text


@TEXT_POSTPROCESSORS.register_module("strip_thinking")
def strip_thinking_postprocess(text):
    if isinstance(text, dict):
        text = str(text.get("prediction", "") or "")
    elif not isinstance(text, str):
        text = str(text)
    return strip_thinking_content(text)


def normalize_answer(answer):
    if not answer:
        return ""
    answer = answer.lower().strip()
    match = re.match(r"^([a-h])[\.\)]\s*\w+", answer)
    if match:
        return match.group(1)
    answer = re.sub(r"^\(([a-h])\)$", r"\1", answer)
    answer = re.sub(r"[.,;:!?]$", "", answer)
    if len(answer) > 1:
        answer = re.sub(r"\b(a|an|the)\b", "", answer)
    return re.sub(r"\s+", " ", answer).strip()


def answers_match(extracted, gold):
    extracted = normalize_answer(extracted)
    gold = normalize_answer(gold)
    if extracted == gold:
        return True
    if (
        (len(extracted) == 1 and extracted.isalpha())
        or (len(gold) == 1 and gold.isalpha())
    ):
        return False
    return len(extracted) > 1 and len(gold) > 1 and (
        gold in extracted or extracted in gold
    )


def extract_answer(judge_response):
    marker = "Extracted Candidate's Answer:"
    if marker in judge_response:
        return judge_response.split(marker, 1)[1].strip()
    return judge_response.strip()


@DICT_POSTPROCESSORS.register_module("judge_answer_extraction")
def judge_answer_extraction_postprocess(
    output, output_path, metric_name="accuracy"
):
    correct_count = 0
    no_answer_count = 0
    multiple_answers_count = 0
    details = []

    for sample_id, item in output.items():
        response = item.get("prediction", "")
        if isinstance(response, list):
            response = response[0] if response else ""
        response = str(response)

        gold = item.get("gold", "")
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        gold = str(gold).strip()

        extracted = extract_answer(response).strip()
        if extracted == "NO_ANSWER":
            is_correct = False
            no_answer_count += 1
        elif extracted == "MULTIPLE_ANSWERS":
            is_correct = False
            multiple_answers_count += 1
        else:
            is_correct = answers_match(extracted, gold)
        correct_count += int(is_correct)
        details.append(
            {
                "id": sample_id,
                "judge_extracted_answer": extracted,
                "gold_answer": gold,
                "is_correct": is_correct,
                "origin_prompt": item.get("origin_prompt", ""),
                "full_judge_response": response,
            }
        )

    total_count = len(details)

    def percentage(count):
        return (100 * count / total_count) if total_count else 0.0

    return {
        metric_name: percentage(correct_count),
        "correct_count": correct_count,
        "total_count": total_count,
        "no_answer_count": no_answer_count,
        "multiple_answers_count": multiple_answers_count,
        "no_answer_rate": percentage(no_answer_count),
        "multiple_answers_rate": percentage(multiple_answers_count),
        "details": details,
    }
