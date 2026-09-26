from causal_evaluation.experiments.storage import append_unique, load_existing_keys


def test_load_existing_keys_empty_for_missing_file(tmp_path):
    assert load_existing_keys(tmp_path / "missing.jsonl", "request_id") == set()


def test_load_existing_keys_reads_written_records(tmp_path):
    path = tmp_path / "responses.jsonl"
    append_unique(path, {"request_id": "a", "value": 1}, "request_id")
    append_unique(path, {"request_id": "b", "value": 2}, "request_id")

    assert load_existing_keys(path, "request_id") == {"a", "b"}


def test_append_unique_skips_duplicate_key(tmp_path):
    path = tmp_path / "responses.jsonl"
    assert append_unique(path, {"request_id": "a", "value": 1}, "request_id") is True
    assert append_unique(path, {"request_id": "a", "value": 2}, "request_id") is False
    assert load_existing_keys(path, "request_id") == {"a"}


def test_append_unique_uses_and_updates_a_preloaded_existing_set(tmp_path):
    path = tmp_path / "responses.jsonl"
    existing = {"a"}

    written = append_unique(path, {"request_id": "b", "value": 1}, "request_id", existing=existing)

    assert written is True
    assert existing == {"a", "b"}
    # Passing `existing` must avoid re-reading the file to decide - "a" was
    # never actually written to disk, only asserted as already-known.
    assert load_existing_keys(path, "request_id") == {"b"}
