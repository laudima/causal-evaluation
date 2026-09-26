"""CLadder commonsense-alignment variants and item-level pairing.

The commonsense and anticommonsense files contain the same formal questions
(story, graph, query type, given probabilities, answer) with one variable
renamed, so their items can be matched one to one. Noncommonsense items use
different parameters and are compared as a group.
"""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable
from pathlib import Path

VARIANTS = ("commonsense", "anticommonsense", "noncommonsense")

# Query types that ask whether X affects or is associated with Y (used for H1b).
RELATIONAL_QUERY_TYPES = frozenset(
    {"correlation", "ate", "ett", "nde", "nie", "det-counterfactual"}
)


def load_variant(raw_dir: str | Path, variant: str) -> list[dict]:
    """Load one CLadder v1 question file as raw records."""
    path = Path(raw_dir) / f"cladder-v1-q-{variant}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_backgrounds(raw_dir: str | Path) -> dict[int, str]:
    """Map meta-model id to its background text, which states the causal graph."""
    path = Path(raw_dir) / "cladder-v1-meta-models.json"
    return {m["model_id"]: m["background"] for m in json.loads(path.read_text(encoding="utf-8"))}


def pair_key(record: dict) -> tuple:
    """Formal content of a question, independent of the variable names."""
    meta = record["meta"]
    suffix = re.search(r"-spec\d+-q\d+$", record["desc_id"]).group(0)
    return (
        meta["story_id"],
        meta["graph_id"],
        meta["query_type"],
        json.dumps(meta["given_info"], sort_keys=True),
        meta.get("polarity"),
        meta.get("treated"),
        meta.get("result"),
        suffix,
    )


def build_pairs(commonsense: list[dict], anticommonsense: list[dict]) -> list[tuple[dict, dict]]:
    """Match commonsense and anticommonsense items one to one.

    Keys that occur more than once in either file are ambiguous and dropped.
    """
    count_c = Counter(map(pair_key, commonsense))
    count_a = Counter(map(pair_key, anticommonsense))
    unique_c = {pair_key(r): r for r in commonsense if count_c[pair_key(r)] == 1}
    pairs = []
    for record in anticommonsense:
        key = pair_key(record)
        match = unique_c.get(key)
        if count_a[key] == 1 and match is not None and match["answer"] == record["answer"]:
            pairs.append((match, record))
    return pairs


def stratified_sample(
    records: list, size: int, seed: int, stratum: Callable[[object], Hashable]
) -> list:
    """Take `size` records spread evenly over strata, reproducibly."""
    groups: dict[Hashable, list] = defaultdict(list)
    for record in records:
        groups[stratum(record)].append(record)
    rng = random.Random(seed)
    keys = sorted(groups, key=str)
    for key in keys:
        rng.shuffle(groups[key])
    selected: list = []
    while len(selected) < size and any(groups[key] for key in keys):
        for key in keys:
            if groups[key] and len(selected) < size:
                selected.append(groups[key].pop())
    return selected


def to_item(record: dict, variant: str, background: str, pair_id: str | None = None) -> dict:
    """Flatten a raw CLadder record into one experimental item."""
    meta = record["meta"]
    return {
        "item_id": f"{variant}-{record['question_id']}",
        "variant": variant,
        "pair_id": pair_id,
        "question_id": record["question_id"],
        "desc_id": record["desc_id"],
        "story_id": meta["story_id"],
        "graph_id": meta["graph_id"],
        "query_type": meta["query_type"],
        "rung": int(meta["rung"]),
        "model_id": meta["model_id"],
        "polarity": meta.get("polarity"),
        "answer": record["answer"].strip().lower(),
        "background": background,
        "given_info": record["given_info"],
        "question": record["question"],
    }


def _rung_answer(record: dict) -> tuple:
    return (record["meta"]["rung"], record["answer"])


def build_variant_sample(raw_dir: str | Path, size_per_variant: int, seed: int) -> list[dict]:
    """Paired commonsense/anticommonsense items plus unpaired noncommonsense items.

    Each variant contributes `size_per_variant` items, stratified by rung and answer.
    """
    backgrounds = load_backgrounds(raw_dir)
    pairs = build_pairs(
        load_variant(raw_dir, "commonsense"), load_variant(raw_dir, "anticommonsense")
    )
    chosen_pairs = stratified_sample(pairs, size_per_variant, seed, lambda p: _rung_answer(p[0]))
    items = []
    for index, (common, anti) in enumerate(chosen_pairs):
        pair_id = f"pair-{index:05d}"
        for record, variant in ((common, "commonsense"), (anti, "anticommonsense")):
            background = backgrounds[record["meta"]["model_id"]]
            items.append(to_item(record, variant, background, pair_id))
    nonsense = stratified_sample(
        load_variant(raw_dir, "noncommonsense"), size_per_variant, seed, _rung_answer
    )
    for record in nonsense:
        items.append(to_item(record, "noncommonsense", backgrounds[record["meta"]["model_id"]]))
    return items


def write_jsonl(items: list[dict], path: str | Path) -> str:
    """Write items as JSONL and return the file's SHA-256."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    target.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
