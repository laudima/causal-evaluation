"""Command-line entry points for the evaluation pipeline."""

import argparse
import json
from glob import glob
from pathlib import Path

from dotenv import load_dotenv

from .config import load_yaml
from .data.loader import load_questions
from .data.sampling import sample_questions
from .evaluation.analysis import evaluate_records, load_evaluations
from .experiments.runner import run_causal_cot_experiment, run_experiment
from .models.providers import build_client
from .viewer.records import build_question_records
from .viewer.render import render_viewer_html


def _write_jsonl(path: str, records: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    """Run a causal evaluation subcommand."""
    load_dotenv()
    parser = argparse.ArgumentParser(prog="causal-evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-data")
    validate.add_argument("--input", default="data/raw/cladder.jsonl")
    sample = subparsers.add_parser("sample-data")
    sample.add_argument("--config", default="configs/experiments.yaml")
    run = subparsers.add_parser("run")
    run.add_argument("--config", default="configs/experiments.yaml")
    run.add_argument("--dry-run", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", default="configs/experiments.yaml")
    report = subparsers.add_parser("generate-reports")
    report.add_argument("--config", default="configs/experiments.yaml")
    report.add_argument(
        "--evaluations-glob",
        default=None,
        help="Glob of evaluations*.jsonl files to combine (e.g. for comparing multiple runs), "
        "instead of the single path in --config.",
    )
    visualize = subparsers.add_parser("visualize")
    visualize.add_argument("--config", default="configs/experiments.yaml")
    visualize.add_argument("--output", default="reports/figures/question_viewer.html")
    args = parser.parse_args()

    if args.command == "validate-data":
        print(f"Validated {len(load_questions(args.input))} questions")
        return
    config = load_yaml(args.config)["experiment"]
    if args.command == "sample-data":
        questions = load_questions(config["input_path"])
        selected = sample_questions(questions, config["sample_size"], config["seed"])
        records = [
            question.model_dump() | {"question_id": question.stable_id()} for question in selected
        ]
        _write_jsonl(config["sample_path"], records)
        print(f"Sampled {len(selected)} questions")
    elif args.command == "run":
        questions = load_questions(config["sample_path"])
        provider = "dry-run" if args.dry_run else config.get("provider", "dry-run")
        client = build_client(provider, model=config["model"])
        prompts = load_yaml("configs/prompts.yaml")["prompts"]
        prompt_strategy = "default" if args.dry_run else config.get("prompt", "default")
        if prompt_strategy == "causal_cot":
            count = run_causal_cot_experiment(
                questions,
                client,
                prompts["causal_cot_step1"],
                prompts["causal_cot_step2"],
                config["model"],
                provider,
                config["responses_path"],
                config["name"],
            )
        else:
            count = run_experiment(
                questions,
                client,
                prompts["default"],
                config["model"],
                provider,
                config["responses_path"],
                config["name"],
            )
        print(f"Stored {count} new responses")
    elif args.command == "evaluate":
        responses = [
            json.loads(line)
            for line in Path(config["responses_path"]).read_text(encoding="utf-8").splitlines()
            if line
        ]
        questions = [
            question.model_dump() | {"question_id": question.stable_id()}
            for question in load_questions(config["sample_path"])
        ]
        frame = evaluate_records(responses, questions)
        Path(config["evaluations_path"]).parent.mkdir(parents=True, exist_ok=True)
        frame.to_json(config["evaluations_path"], orient="records", lines=True)
        print(f"Evaluated {len(frame)} responses")
    elif args.command == "generate-reports":
        if args.evaluations_glob:
            frame = load_evaluations(sorted(glob(args.evaluations_glob)))
        else:
            frame = load_evaluations([config["evaluations_path"]])
        frame.groupby(["model", "provider", "prompt_strategy", "causal_level"], dropna=False)[
            "correct"
        ].mean().to_csv("reports/tables/accuracy_by_level.csv")
        print("Generated reports/tables/accuracy_by_level.csv")
    elif args.command == "visualize":
        records = build_question_records(
            config["sample_path"],
            config["responses_path"],
            config["evaluations_path"],
            model=config["model"],
        )
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_viewer_html(records), encoding="utf-8")
        print(f"Wrote viewer for {len(records)} questions to {destination}")


if __name__ == "__main__":
    main()
