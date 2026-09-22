"""Benchmark groups reported for QVAC Genesis III."""

import copy

from genesis_iii_eval import (
    GenesisIIIGenericLLMEvaluator,
    judge_answer_extraction_postprocess,  # noqa: F401
    strip_thinking_postprocess,  # noqa: F401
)
from opencompass.datasets import ARCDataset, GPQADataset, MMLUDataset
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


MMLU_STEM_SUBSETS = [
    "astronomy",
    "electrical_engineering",
    "high_school_geography",
    "college_biology",
    "high_school_biology",
    "college_medicine",
    "professional_medicine",
    "college_mathematics",
    "high_school_mathematics",
    "college_physics",
    "high_school_physics",
    "conceptual_physics",
    "college_chemistry",
    "high_school_chemistry",
    "college_computer_science",
    "high_school_computer_science",
    "machine_learning",
    "high_school_statistics",
    "econometrics",
]

SYSTEM_PROMPT = (
    "You are a helpful assistant who extracts the final answer from models' "
    "outputs."
)


def extraction_prompt(benchmark, original_question):
    """Build the exact LLM-as-a-parser instruction used in the paper runs."""
    return f"""
    You are an expert answer extractor. Your ONLY job is to extract the final answer from the candidate's response.

    CRITICAL INSTRUCTIONS - READ CAREFULLY:
    - DO NOT solve the question yourself
    - DO NOT generate a new answer
    - DO NOT think about what the correct answer should be
    - DO NOT evaluate whether the candidate's answer is right or wrong
    - ONLY extract what the candidate actually wrote as their final answer

    Your task is purely extraction, not generation or evaluation.

    Here are the extraction guidelines for {benchmark} multiple choice questions:
    1. Look for the candidate's final answer in their response. This should be one of: A, B, C, or D
       - Look for explicit statements like "ANSWER: A", "The answer is B", "Final answer: C"
       - Look for \\boxed{{{{A}}}} format (extract what's inside the braces)
       - Look for standalone letters A, B, C, or D in their conclusion
       - The final choice they settle on in their reasoning

    2. If the candidate's response contains multiple different answers:
       - If they state one answer but then correct/fix themselves, extract the corrected/final answer they provided
       - If they state multiple different answers without choosing one or correcting themselves, return "MULTIPLE_ANSWERS"

    3. If you cannot clearly identify a single letter choice (A, B, C, or D) in the response, return "NO_ANSWER"

    4. If the candidate generates new questions:
        - DO NOT extract the answer from the new generated questions
        - ONLY extract the answer from the original question

    RESPONSE FORMAT:
    First, provide a brief explanation of why you are extracting that particular answer (what indicators you found in the candidate's response).
    Then, provide the extracted answer as a single letter.

    Use this exact format:

    Extraction Reasoning: [Brief explanation of what indicators led you to extract this answer from the candidate's response]

    Extracted Candidate's Answer: [A single letter: A, B, C, or D. Use MULTIPLE_ANSWERS if the candidate provided multiple different answers, or NO_ANSWER if no clear answer was found.]

    <Original Question Begin>: \n{original_question}\n<Original Question End>\n\n
    <Candidate's Response Begin>: \n{{prediction}}\n<Candidate's Response End>\n\n
""".strip()


def make_dataset(*, abbr, dataset_type, path, name, reader_cfg, query, extractor):
    dataset_cfg = dict(
        type=dataset_type,
        path=path,
        name=name,
        reader_cfg=reader_cfg,
    )
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[dict(role="HUMAN", prompt=query)]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    eval_cfg = dict(
        evaluator=dict(
            type=GenesisIIIGenericLLMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role="SYSTEM",
                            fallback_role="HUMAN",
                            prompt=SYSTEM_PROMPT,
                        )
                    ],
                    round=[dict(role="HUMAN", prompt=extractor)],
                ),
            ),
            dataset_cfg=dataset_cfg,
            pred_postprocessor=dict(type="strip_thinking"),
            dict_postprocessor=dict(
                type="judge_answer_extraction", metric_name="score"
            ),
        ),
        pred_role="BOT",
    )
    return dict(
        abbr=abbr,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
        **dataset_cfg,
    )


MMLU_READER = dict(
    input_columns=["input", "A", "B", "C", "D"],
    output_column="target",
    train_split="test",
)
MMLU_QUERY = (
    "Question: {input}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nAnswer:"
)
MMLU_EXTRACTOR = extraction_prompt(
    "MMLU",
    "Question: {input}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nAnswer: ",
)

ARC_READER = dict(
    input_columns=["question", "textA", "textB", "textC", "textD"],
    output_column="answerKey",
)
ARC_QUERY = (
    "Question: {question}\n\nA. {textA}\nB. {textB}\nC. {textC}\n"
    "D. {textD}\n\nAnswer:"
)
ARC_EXTRACTOR = extraction_prompt(
    "ARC (AI2 Reasoning Challenge)",
    "Question: {question}\n\nA. {textA}\nB. {textB}\nC. {textC}\n"
    "D. {textD}\n\nAnswer: ",
)

GPQA_READER = dict(
    input_columns=["question", "A", "B", "C", "D"],
    output_column="answer",
)
GPQA_QUERY = (
    "Question: {question}\n\nOptions:\nA) {A}\nB) {B}\nC) {C}\n"
    "D) {D}\n\nAnswer:"
)
# Preserve the "\Options:" typo from the paper-evaluation parser prompt.
GPQA_EXTRACTOR = extraction_prompt(
    "GPQA (Graduate-Level Google-Proof Q&A)",
    "Question: {question}\n\\Options:\nA) {A}\nB) {B}\nC) {C}\n"
    "D) {D}\n\nAnswer: ",
)

mmlu_genesis_stem_datasets = [
    make_dataset(
        abbr=f"lukaemon_mmlu_{subset}",
        dataset_type=MMLUDataset,
        path="opencompass/mmlu",
        name=subset,
        reader_cfg=MMLU_READER,
        query=MMLU_QUERY,
        extractor=MMLU_EXTRACTOR,
    )
    for subset in MMLU_STEM_SUBSETS
]
arc_challenge_datasets = [
    make_dataset(
        abbr="ARC-c",
        dataset_type=ARCDataset,
        path="opencompass/ai2_arc-dev",
        name="ARC-Challenge",
        reader_cfg=ARC_READER,
        query=ARC_QUERY,
        extractor=ARC_EXTRACTOR,
    )
]
arc_easy_datasets = [
    make_dataset(
        abbr="ARC-e",
        dataset_type=ARCDataset,
        path="opencompass/ai2_arc-easy-dev",
        name="ARC-Easy",
        reader_cfg=ARC_READER,
        query=ARC_QUERY,
        extractor=ARC_EXTRACTOR,
    )
]
gpqa_diamond_datasets = [
    make_dataset(
        abbr="GPQA_diamond",
        dataset_type=GPQADataset,
        path="./data/gpqa/",
        name="gpqa_diamond.csv",
        reader_cfg=GPQA_READER,
        query=GPQA_QUERY,
        extractor=GPQA_EXTRACTOR,
    )
]

genesis_iii_datasets = (
    mmlu_genesis_stem_datasets
    + arc_challenge_datasets
    + arc_easy_datasets
    + gpqa_diamond_datasets
)

smoke_datasets = [copy.deepcopy(arc_easy_datasets[0])]
smoke_datasets[0]["abbr"] = "ARC-e-smoke"
smoke_datasets[0]["reader_cfg"]["test_range"] = "[0:2]"
smoke_datasets[0]["eval_cfg"]["evaluator"]["dataset_cfg"]["reader_cfg"][
    "test_range"
] = "[0:2]"
