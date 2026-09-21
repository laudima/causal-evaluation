# Graph Report - causal-evaluation  (2026-09-03)

## Corpus Check
- Corpus is ~1,482 words - fits in a single context window. You may not need a graph.

## Summary
- 102 nodes · 128 edges · 13 communities (7 shown, 6 thin omitted)
- Extraction: 95% EXTRACTED · 4% INFERRED · 1% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.87)
- Token cost: 54,145 input · 0 output

## Community Hubs (Navigation)
- CLI & Config Loading
- Project Docs & Pipeline Concepts
- Evaluation & Metrics
- Experiment Runner & Storage
- Model Provider Interface
- Failure Graph Construction
- Data Validation & Sampling Tests
- Evaluation Package Init
- Experiments Package Init
- Failure Graph Package Init
- Project Package Init
- Models Package Init
- Package Root

## God Nodes (most connected - your core abstractions)
1. `main()` - 9 edges
2. `run_experiment()` - 8 edges
3. `evaluate_records()` - 7 edges
4. `build_failure_graph()` - 7 edges
5. `ModelResponse` - 7 edges
6. `load_yaml()` - 6 edges
7. `score_response()` - 6 edges
8. `append_unique()` - 6 edges
9. `save_graph()` - 6 edges
10. `LLMClient` - 6 edges

## Surprising Connections (you probably didn't know these)
- `validate-data CLI command` --references--> `cladder-baseline experiment config`  [AMBIGUOUS]
  README.md → configs/experiments.yaml
- `test_evaluation_preserves_metadata()` --calls--> `evaluate_records()`  [EXTRACTED]
  tests/test_evaluation.py → src/causal_evaluation/evaluation/analysis.py
- `test_scoring_normalizes_case_and_whitespace()` --calls--> `score_response()`  [EXTRACTED]
  tests/test_evaluation.py → src/causal_evaluation/evaluation/metrics.py
- `test_failure_graph_counts_edges()` --calls--> `build_failure_graph()`  [EXTRACTED]
  tests/test_failure_graph.py → src/causal_evaluation/failure_graph/graph.py
- `cladder-baseline experiment config` --references--> `CLadder Benchmark`  [EXTRACTED]
  configs/experiments.yaml → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Pipeline Configuration Files** — configs_experiments_cladder_baseline, configs_models_dry_run, configs_prompts_default [EXTRACTED 1.00]
- **Causal Evaluation CLI Pipeline Stages** — readme_validate_data_command, readme_sample_data_command, readme_run_command, readme_evaluate_command, readme_build_failure_graph_command, readme_generate_reports_command [EXTRACTED 1.00]

## Communities (13 total, 6 thin omitted)

### Community 0 - "CLI & Config Loading"
Cohesion: 0.13
Nodes (17): main(), Command-line entry points for the evaluation pipeline., Run a causal evaluation subcommand., _write_jsonl(), load_yaml(), Any, Path, Configuration loading helpers. (+9 more)

### Community 1 - "Project Docs & Pipeline Concepts"
Cohesion: 0.12
Nodes (18): cladder-baseline experiment config, dry-run model config, default prompt template, Responsible API Study Practices, build-failure-graph CLI command, Causal Evaluation Project, CLadder Benchmark, Expected Dataset Schema (+10 more)

### Community 2 - "Evaluation & Metrics"
Cohesion: 0.23
Nodes (10): evaluate_records(), DataFrame, Evaluation result construction and grouping., Join raw responses to benchmark answers and produce normalized results., normalize_answer(), Return whether a model answer matches the expected answer., Normalize an answer for exact-match evaluation., score_response() (+2 more)

### Community 3 - "Experiment Runner & Storage"
Cohesion: 0.21
Nodes (10): Render a configured prompt, optionally adding a dry-run answer marker., render_prompt(), Resumable experiment runner., Query each question once and persist raw responses as JSONL., run_experiment(), append_unique(), Any, Path (+2 more)

### Community 4 - "Model Provider Interface"
Cohesion: 0.20
Nodes (9): BaseModel, Protocol, LLMClient, ModelResponse, Provider protocol and response schema., Raw provider response retained for auditability., Minimal interface implemented by every provider., DryRunClient (+1 more)

### Community 5 - "Failure Graph Construction"
Cohesion: 0.29
Nodes (6): build_failure_graph(), DataFrame, DiGraph, Directed graph of recurring failure patterns., Connect metadata categories to observed incorrect outcomes., test_failure_graph_counts_edges()

### Community 6 - "Data Validation & Sampling Tests"
Cohesion: 0.33
Nodes (3): BenchmarkQuestion, test_malformed_question_rejected(), test_sampling_is_reproducible_and_stratified()

## Ambiguous Edges - Review These
- `validate-data CLI command` → `cladder-baseline experiment config`  [AMBIGUOUS]
  README.md · relation: references

## Knowledge Gaps
- **4 isolated node(s):** `causal-evaluation`, `generate-reports CLI command`, `Quality Checks (pytest, ruff)`, `default prompt template`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 54 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `validate-data CLI command` and `cladder-baseline experiment config`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `run_experiment()` connect `Experiment Runner & Storage` to `CLI & Config Loading`, `Model Provider Interface`, `Data Validation & Sampling Tests`?**
  _High betweenness centrality (0.145) - this node is a cross-community bridge._
- **Why does `main()` connect `CLI & Config Loading` to `Evaluation & Metrics`, `Experiment Runner & Storage`, `Failure Graph Construction`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `evaluate_records()` connect `Evaluation & Metrics` to `CLI & Config Loading`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **What connects `causal-evaluation`, `generate-reports CLI command`, `Quality Checks (pytest, ruff)` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `CLI & Config Loading` be split into smaller, more focused modules?**
  _Cohesion score 0.12857142857142856 - nodes in this community are weakly interconnected._
- **Should `Project Docs & Pipeline Concepts` be split into smaller, more focused modules?**
  _Cohesion score 0.12418300653594772 - nodes in this community are weakly interconnected._