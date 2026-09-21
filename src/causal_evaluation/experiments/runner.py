"""Resumable experiment runner."""

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from ..data.validation import BenchmarkQuestion
from ..models.base import LLMClient
from .prompts import render_causal_cot_step1, render_causal_cot_step2, render_prompt
from .schema import ResponseRecord
from .storage import append_unique


def run_experiment(
    questions: list[BenchmarkQuestion],
    client: LLMClient,
    template: str,
    model: str,
    provider: str,
    output_path: str,
    run_id: str,
) -> int:
    """Query each question once and persist raw responses as JSONL."""
    written = 0
    for question in questions:
        question_id = question.stable_id()
        request_id = str(uuid5(NAMESPACE_URL, f"{run_id}:{model}:{question_id}"))
        prompt = render_prompt(
            template, question.question, question.answer if model == "dry-run" else None
        )
        response = client.complete(prompt)
        record = ResponseRecord(
            request_id=request_id,
            run_id=run_id,
            question_id=question_id,
            model=model,
            provider=provider,
            prompt_strategy="default",
            prompt=prompt,
            response=response,
            created_at=datetime.now(UTC),
        )
        written += append_unique(output_path, record.model_dump(mode="json"), "request_id")
    return written


def run_causal_cot_experiment(
    questions: list[BenchmarkQuestion],
    client: LLMClient,
    step1_template: str,
    step2_template: str,
    model: str,
    provider: str,
    output_path: str,
    run_id: str,
) -> int:
    """Query each question with the two-turn CausalCoT prompt, persisting the final answer."""
    written = 0
    for question in questions:
        question_id = question.stable_id()
        request_id = str(uuid5(NAMESPACE_URL, f"{run_id}:{model}:{question_id}"))
        step1_prompt = render_causal_cot_step1(
            step1_template, question.given_info, question.question
        )
        reasoning = client.complete(step1_prompt)
        step2_prompt = render_causal_cot_step2(
            step2_template, question.given_info, question.question, reasoning.text
        )
        final = client.complete(step2_prompt)
        record = ResponseRecord(
            request_id=request_id,
            run_id=run_id,
            question_id=question_id,
            model=model,
            provider=provider,
            prompt_strategy="causal_cot",
            prompt=step2_prompt,
            reasoning_prompt=step1_prompt,
            reasoning=reasoning,
            response=final,
            created_at=datetime.now(UTC),
        )
        written += append_unique(output_path, record.model_dump(mode="json"), "request_id")
    return written
