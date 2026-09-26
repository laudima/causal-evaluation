from causal_evaluation.data.variants import build_pairs, pair_key, stratified_sample, to_item


def _record(qid, story, answer="yes", model_id=0, rung=2, given=(0.1, 0.2), suffix="-spec0-q0"):
    return {
        "question_id": qid,
        "desc_id": f"{story}-chain-ate-model{model_id}{suffix}",
        "given_info": "text",
        "question": f"Q {qid}",
        "answer": answer,
        "meta": {
            "story_id": story,
            "graph_id": "chain",
            "query_type": "ate",
            "rung": rung,
            "model_id": model_id,
            "given_info": {"p(Y | X)": list(given)},
            "polarity": True,
            "treated": True,
            "result": True,
        },
    }


def test_pair_key_ignores_model_number_and_names():
    common = _record(0, "smoking", model_id=3)
    anti = _record(0, "smoking", model_id=1113)
    anti["question"] = "renamed variable"
    assert pair_key(common) == pair_key(anti)


def test_build_pairs_drops_ambiguous_and_mismatched():
    common = [_record(0, "a"), _record(1, "b"), _record(2, "c"), _record(3, "c")]
    anti = [_record(10, "a"), _record(11, "b", answer="no"), _record(12, "c")]
    pairs = build_pairs(common, anti)
    assert [(c["question_id"], a["question_id"]) for c, a in pairs] == [(0, 10)]


def test_stratified_sample_is_balanced_and_reproducible():
    records = [{"s": i % 3, "i": i} for i in range(30)]
    first = stratified_sample(records, 9, 7, lambda r: r["s"])
    assert first == stratified_sample(records, 9, 7, lambda r: r["s"])
    assert sorted(r["s"] for r in first) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_to_item_prefixes_variant_in_id():
    item = to_item(_record(5, "a", answer="Yes "), "anticommonsense", "bg", "pair-00001")
    assert item["item_id"] == "anticommonsense-5"
    assert item["answer"] == "yes"
    assert item["pair_id"] == "pair-00001"
