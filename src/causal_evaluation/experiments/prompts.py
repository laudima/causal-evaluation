"""Prompt rendering."""


def render_prompt(template: str, question: str, answer: str | None = None) -> str:
    """Render a configured prompt, optionally adding a dry-run answer marker."""
    rendered = template.format(question=question)
    return f"{rendered}\n[DRY_RUN_ANSWER={answer}]" if answer is not None else rendered


def render_causal_cot_step1(template: str, given_info: str, question: str) -> str:
    """Render the first-turn CausalCoT prompt (step-by-step causal reasoning)."""
    return template.format(given_info=given_info, question=question)


def render_causal_cot_step2(template: str, given_info: str, question: str, reasoning: str) -> str:
    """Render the second-turn CausalCoT prompt, replaying the reasoning to get a final answer."""
    return template.format(given_info=given_info, question=question, reasoning=reasoning)
