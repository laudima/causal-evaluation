"""Yes/No answer probabilities from open-weight causal language models.

One forward pass per item: the answer is read from the next-token distribution
instead of being generated and parsed. torch and transformers are imported
lazily so the rest of the package does not depend on them.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

INSTRUCTION = "Answer with Yes or No only."
YES_VARIANTS = ("Yes", " Yes", "yes", " yes", "YES")
NO_VARIANTS = ("No", " No", "no", " no", "NO")


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


def score_batch(model, tokenizer, items: list[dict], yes_ids: list[int], no_ids: list[int]):
    """P(Yes), probability mass on Yes/No tokens, and the top next token, per item."""
    import torch

    rendered = [render_prompt(tokenizer, item_text(item)) for item in items]
    add_special = rendered[0][1]
    encoded = tokenizer(
        [prompt for prompt, _ in rendered],
        return_tensors="pt",
        padding=True,
        add_special_tokens=add_special,
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
        for i in range(len(items))
    ]


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


def done_item_ids(path: str | Path, model_name: str) -> set[str]:
    """Item ids already scored for this model (for resuming)."""
    target = Path(path)
    if not target.exists():
        return set()
    with target.open(encoding="utf-8") as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        return {row["item_id"] for row in rows if row["model"] == model_name}


def run_model(
    items: list[dict],
    model_name: str,
    out_path: str | Path,
    batch_size: int = 8,
    load_in_4bit: bool = False,
    dtype: str = "float32",
    progress: bool = True,
) -> int:
    """Score all pending items for one model, appending to `out_path`. Returns items scored."""
    import transformers

    pending = [item for item in items if item["item_id"] not in done_item_ids(out_path, model_name)]
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
            for item, scores in zip(
                batch, score_batch(model, tokenizer, batch, yes_ids, no_ids), strict=True
            ):
                row = {
                    "item_id": item["item_id"],
                    "model": model_name,
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
