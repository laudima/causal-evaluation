"""Yes/No answer probabilities from open-weight causal language models.

The answer is read from the next-token distribution instead of being generated
and parsed. Two prompt strategies:

- ``direct``: one user turn ending in "Answer with Yes or No only."
- ``causal_cot``: the model first writes the six CausalCoT steps of Jin et al.
  (2023); its reasoning is kept in the conversation and a second turn asks for
  just Yes or No, whose probability is read from the first answer token.

torch and transformers are imported lazily so the rest of the package does not
depend on them.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

INSTRUCTION = "Answer with Yes or No only."
YES_VARIANTS = ("Yes", " Yes", "yes", " yes", "YES")
NO_VARIANTS = ("No", " No", "no", " no", "NO")

COT_GUIDANCE = "\n\n".join(
    [
        ("Guidance: Address the question by following the steps below:"),
        (
            "Step 1) Extract the causal graph: Identify the causal graph that depicts the "
            "relationships in the scenario. The diagram should simply consist of edges "
            'denoted in "var1 -> var2" format, separated by commas.'
        ),
        (
            "Step 2) Determine the query type: Identify the type of query implied by the main "
            'question. Choices include "marginal probability", "conditional probability", '
            '"explaining away effect", "backdoor adjustment set", "average treatment effect", '
            '"collider bias", "normal counterfactual question", "average treatment effect on '
            'treated", "natural direct effect" or "natural indirect effect". Your answer '
            "should only be a term from the list above, enclosed in quotation marks."
        ),
        (
            "Step 3) Formalize the query: Translate the query into its formal mathematical "
            'expression based on its type, utilizing the "do(·)" notation or counterfactual '
            "notations as needed."
        ),
        (
            "Step 4) Gather all relevant data: Extract all the available data. Your answer "
            "should contain nothing but marginal probabilities and conditional probabilities "
            'in the form "P(...)=..." or "P(...|...)=...", each probability being separated '
            "by a semicolon. Stick to the previously mentioned denotations for the variables."
        ),
        (
            "Step 5) Deduce the estimand using causal inference: Given all the information "
            "above, deduce the estimand using skills such as do-calculus, counterfactual "
            "prediction, and the basics of probabilities. Answer step by step."
        ),
        (
            "Step 6) Calculate the estimand: Insert the relevant data in Step 4 into the "
            "estimand, perform basic arithmetic calculations, and derive the final answer. "
            "There is an identifiable answer. Answer step by step."
        ),
    ]
)
COT_FINAL_QUESTION = (
    "Based on all the reasoning above, answer the initial question with just Yes or No."
)
PROMPT_STRATEGIES = ("direct", "causal_cot")


def item_text(item: dict) -> str:
    """User message: background (states the graph), given data, question, instruction."""
    return f"{item['background']}\n{item['given_info']}\n{item['question']}\n{INSTRUCTION}"


def render_prompt(tokenizer, text: str) -> tuple[str, bool]:
    """Apply the chat template if the model has one.

    Returns the prompt and whether the tokenizer should add special tokens
    (chat templates already include them).
    """
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        return prompt, False
    return f"{text}\nAnswer:", True


def cot_messages(item: dict, reasoning: str | None = None) -> list[dict]:
    """CausalCoT conversation; with `reasoning`, adds it and the final Yes/No turn."""
    question = f"{item['background']}\n{item['given_info']}\n{item['question']}"
    messages = [{"role": "user", "content": f"{question}\n\n{COT_GUIDANCE}"}]
    if reasoning is not None:
        messages += [
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": COT_FINAL_QUESTION},
        ]
    return messages


def render_messages(tokenizer, messages: list[dict]) -> tuple[str, bool]:
    """Multi-turn version of `render_prompt`."""
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt, False
    turns = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
    return f"{turns}\n\nAssistant:", True


def answer_token_ids(tokenizer, variants: tuple[str, ...]) -> list[int]:
    """Ids of the variants that are a single token; raises if there are none."""
    ids = set()
    for variant in variants:
        tokens = tokenizer.encode(variant, add_special_tokens=False)
        if len(tokens) == 1:
            ids.add(tokens[0])
    if not ids:
        raise ValueError(f"No single-token form among {variants} for this tokenizer")
    return sorted(ids)


def score_prompts(model, tokenizer, prompts: list[str], add_special: bool, yes_ids, no_ids):
    """P(Yes), probability mass on Yes/No tokens, and the top next token, per rendered prompt."""
    import torch

    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, add_special_tokens=add_special
    ).to(model.device)
    # Explicit positions so left padding does not shift them.
    positions = (encoded["attention_mask"].cumsum(dim=-1) - 1).clamp(min=0)
    with torch.inference_mode():
        try:
            output = model(**encoded, position_ids=positions, logits_to_keep=1)
        except TypeError:
            output = model(**encoded, position_ids=positions)
    logp = torch.log_softmax(output.logits[:, -1, :].float(), dim=-1)
    log_yes = torch.logsumexp(logp[:, yes_ids], dim=-1)
    log_no = torch.logsumexp(logp[:, no_ids], dim=-1)
    p_yes = torch.sigmoid(log_yes - log_no)
    mass = log_yes.exp() + log_no.exp()
    top = logp.argmax(dim=-1)
    return [
        {
            "p_yes": float(p_yes[i]),
            "answer_mass": float(mass[i]),
            "top_token": tokenizer.decode([int(top[i])]),
            "prompt_tokens": int(encoded["attention_mask"][i].sum()),
        }
        for i in range(len(prompts))
    ]


def score_batch(model, tokenizer, items: list[dict], yes_ids: list[int], no_ids: list[int]):
    """Direct prompt: score each item from a single user turn."""
    rendered = [render_prompt(tokenizer, item_text(item)) for item in items]
    return score_prompts(
        model, tokenizer, [p for p, _ in rendered], rendered[0][1], yes_ids, no_ids
    )


def _eos_ids(model, tokenizer) -> set[int]:
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, list) else [ids] if ids is not None else [])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def generate_reasoning(model, tokenizer, items: list[dict], max_new_tokens: int):
    """Greedy CausalCoT reasoning per item: (text, truncated)."""
    import torch

    rendered = [render_messages(tokenizer, cot_messages(item)) for item in items]
    encoded = tokenizer(
        [p for p, _ in rendered],
        return_tensors="pt",
        padding=True,
        add_special_tokens=rendered[0][1],
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[:, encoded["input_ids"].shape[1] :]
    eos = _eos_ids(model, tokenizer)
    results = []
    for row in new_tokens.tolist():
        finished = any(token in eos for token in row)
        text = tokenizer.decode(row, skip_special_tokens=True).strip()
        results.append((text, not finished))
    return results


def score_cot_batch(
    model, tokenizer, items: list[dict], yes_ids, no_ids, max_new_tokens: int = 512
):
    """CausalCoT prompt: generate reasoning, then score the final Yes/No turn."""
    reasoning = generate_reasoning(model, tokenizer, items, max_new_tokens)
    rendered = [
        render_messages(tokenizer, cot_messages(item, text))
        for item, (text, _) in zip(items, reasoning, strict=True)
    ]
    scores = score_prompts(
        model, tokenizer, [p for p, _ in rendered], rendered[0][1], yes_ids, no_ids
    )
    for score, (text, truncated) in zip(scores, reasoning, strict=True):
        score["reasoning"] = text
        score["reasoning_truncated"] = truncated
    return scores


def load_model(model_name: str, load_in_4bit: bool = False, dtype: str = "float32"):
    """Load tokenizer and model for left-padded batched scoring.

    float32 is the default because half precision (bf16 on MPS in particular)
    makes P(Yes) depend on how items are batched.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "device_map": "auto",
        "dtype": dtype if dtype == "auto" else getattr(torch, dtype),
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def done_item_ids(path: str | Path, model_name: str, prompt_strategy: str = "direct") -> set[str]:
    """Item ids already scored for this model and prompt strategy (for resuming)."""
    target = Path(path)
    if not target.exists():
        return set()
    with target.open(encoding="utf-8") as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        return {
            row["item_id"]
            for row in rows
            if row["model"] == model_name
            and row.get("prompt_strategy", "direct") == prompt_strategy
        }


def run_model(
    items: list[dict],
    model_name: str,
    out_path: str | Path,
    batch_size: int = 8,
    load_in_4bit: bool = False,
    dtype: str = "float32",
    progress: bool = True,
    prompt_strategy: str = "direct",
    max_new_tokens: int = 512,
) -> int:
    """Score all pending items for one model, appending to `out_path`. Returns items scored."""
    import transformers

    if prompt_strategy not in PROMPT_STRATEGIES:
        raise ValueError(f"prompt_strategy must be one of {PROMPT_STRATEGIES}")
    done = done_item_ids(out_path, model_name, prompt_strategy)
    pending = [item for item in items if item["item_id"] not in done]
    if not pending:
        return 0
    model, tokenizer = load_model(model_name, load_in_4bit, dtype)
    yes_ids = answer_token_ids(tokenizer, YES_VARIANTS)
    no_ids = answer_token_ids(tokenizer, NO_VARIANTS)
    revision = getattr(model.config, "_commit_hash", None)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    batches = range(0, len(pending), batch_size)
    if progress:
        from tqdm.auto import tqdm

        batches = tqdm(batches, desc=model_name)
    with target.open("a", encoding="utf-8") as handle:
        for start in batches:
            batch = pending[start : start + batch_size]
            if prompt_strategy == "direct":
                batch_scores = score_batch(model, tokenizer, batch, yes_ids, no_ids)
            else:
                batch_scores = score_cot_batch(
                    model, tokenizer, batch, yes_ids, no_ids, max_new_tokens
                )
            for item, scores in zip(batch, batch_scores, strict=True):
                row = {
                    "item_id": item["item_id"],
                    "model": model_name,
                    "prompt_strategy": prompt_strategy,
                    "max_new_tokens": max_new_tokens if prompt_strategy == "causal_cot" else None,
                    "model_revision": revision,
                    "load_in_4bit": load_in_4bit,
                    "dtype": str(model.dtype),
                    "transformers_version": transformers.__version__,
                    **scores,
                    "created_at": datetime.now(UTC).isoformat(),
                }
                handle.write(json.dumps(row) + "\n")
            handle.flush()
    return len(pending)


def check_batch_invariance(
    model_name: str, items: list[dict], load_in_4bit: bool = False, dtype: str = "float32"
) -> float:
    """Largest |P(Yes) batched - P(Yes) alone| over `items`; should be close to 0."""
    model, tokenizer = load_model(model_name, load_in_4bit, dtype)
    yes_ids = answer_token_ids(tokenizer, YES_VARIANTS)
    no_ids = answer_token_ids(tokenizer, NO_VARIANTS)
    together = score_batch(model, tokenizer, items, yes_ids, no_ids)
    alone = [score_batch(model, tokenizer, [item], yes_ids, no_ids)[0] for item in items]
    return max(abs(a["p_yes"] - b["p_yes"]) for a, b in zip(together, alone, strict=True))
