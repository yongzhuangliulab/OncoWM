"""Two-stage M-JEPA training entry point."""

import argparse
import logging
import time
from pathlib import Path

import yaml


LOGGER = logging.getLogger(__name__)


def run_task(config_path, task, run_name, status_dir, force=False):
    """Run one training stage and create a completion marker on success."""
    status_dir.mkdir(parents=True, exist_ok=True)
    done_file = status_dir / f"{run_name}_{task}.done"

    if done_file.exists() and not force:
        LOGGER.info("Task already completed; remove %s or pass --force to rerun.", done_file)
        return

    LOGGER.info("Loading config: %s", config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    if not isinstance(params, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")

    params["task_mode"] = task
    params["run_name"] = run_name

    # Import heavy ML dependencies only when a task is actually launched. This
    # keeps `python main.py --help` usable before the environment is installed.
    from src.train import main as train_main

    start_time = time.time()
    train_main(params)
    elapsed_time = time.time() - start_time

    done_file.write_text(
        f"Task: {task}\nRun Name: {run_name}\nCompleted in {elapsed_time:.2f} seconds.\n",
        encoding="utf-8",
    )
    LOGGER.info("Task completed in %.2f seconds; marker: %s", elapsed_time, done_file)


def parse_args():
    parser = argparse.ArgumentParser(description="M-JEPA alignment and prognosis training")
    parser.add_argument("--fname", type=Path, required=True, help="Training YAML path")
    parser.add_argument(
        "--task",
        choices=["align", "prognosis", "both"],
        required=True,
        help="Run alignment, prognosis, or both stages in order",
    )
    parser.add_argument(
        "--run-name",
        "--run_name",
        dest="run_name",
        required=True,
        help="Experiment name, for example BRCA_OS",
    )
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=Path("outputs/train_status"),
        help="Directory for successful-stage .done markers",
    )
    parser.add_argument("--force", action="store_true", help="Ignore an existing .done marker")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    tasks = ["align", "prognosis"] if args.task == "both" else [args.task]
    for task in tasks:
        run_task(args.fname, task, args.run_name, args.status_dir, force=args.force)


if __name__ == "__main__":
    main()
