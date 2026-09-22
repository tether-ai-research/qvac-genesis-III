from genesis_iii_eval.postprocessors import (
    judge_answer_extraction_postprocess,
    strip_thinking_postprocess,
)


def test_strip_thinking_keeps_final_answer():
    assert strip_thinking_postprocess("<think>work</think>\nB") == "B"


def test_unclosed_thinking_is_not_judged():
    assert strip_thinking_postprocess("<think>unfinished") == "NO_ANSWER"


def test_parser_aggregation_contract():
    output = {
        "0": {"prediction": "Extracted Candidate's Answer: A", "gold": "A"},
        "1": {
            "prediction": "Extracted Candidate's Answer: NO_ANSWER",
            "gold": "B",
        },
        "2": {
            "prediction": "Extracted Candidate's Answer: MULTIPLE_ANSWERS",
            "gold": "C",
        },
    }
    result = judge_answer_extraction_postprocess(output, "unused")
    assert result["accuracy"] == 100 / 3
    assert result["no_answer_count"] == 1
    assert result["multiple_answers_count"] == 1
