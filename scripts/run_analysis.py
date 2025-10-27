"""End-to-end pipeline that trains an OLS model and generates reports."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src.models.ols import OLSResult, run_ols_pipeline
from src.reports import visualize


DEFAULT_DATASET = Path("baseball_data/basic.csv")
DEFAULT_TEMPLATE = Path("reports/season_summary.md.j2")
DEFAULT_OUTPUT_DIR = Path("reports/generated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseball OLS analysis pipeline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the CSV dataset used for modelling.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="WAR",
        help="Name of the target column to predict.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma separated list of feature columns to include. Defaults to all numeric columns.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for the test split.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross validation splits. Set to 0 to disable.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for train/test splitting and CV shuffling.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Path to the Jinja2 template for the season summary report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated artefacts are stored.",
    )
    return parser.parse_args()


def determine_features(df: pd.DataFrame, target: str, explicit: Optional[str]) -> List[str]:
    if explicit:
        return [col.strip() for col in explicit.split(",") if col.strip()]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [col for col in numeric_cols if col != target]


def load_template(template_path: Path) -> Environment:
    env = Environment(loader=FileSystemLoader(template_path.parent))
    return env


def render_report(template_path: Path, context: dict, output_path: Path) -> None:
    env = load_template(template_path)
    template = env.get_template(template_path.name)
    output_path.write_text(template.render(**context), encoding="utf-8")


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.dataset)
    feature_names = determine_features(data, args.target, args.features)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result: OLSResult = run_ols_pipeline(
        data,
        target_column=args.target,
        feature_columns=feature_names,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_splits=args.cv_splits,
    )

    predictions = result.model.predict(data[feature_names])

    correlation_fig = visualize.plot_correlation_matrix(data[feature_names + [args.target]])
    correlation_path = output_dir / "correlation_matrix.png"
    save_figure(correlation_fig, correlation_path)

    residual_fig = visualize.plot_residuals(data[args.target], predictions)
    residual_path = output_dir / "residual_plot.png"
    save_figure(residual_fig, residual_path)

    feature_fig = visualize.plot_feature_importance(result.model, feature_names)
    feature_path = output_dir / "feature_importance.png"
    save_figure(feature_fig, feature_path)

    context = {
        "generated_on": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "data_source": str(args.dataset),
        "sample_count": int(len(data)),
        "feature_count": len(feature_names),
        "metrics": result.metrics,
        "cv_scores": result.cross_validation_scores,
        "correlation_path": str(correlation_path),
        "residuals_path": str(residual_path),
        "feature_importance_path": str(feature_path),
        "top_features": feature_names[:5],
        "notes": None,
    }

    summary_path = output_dir / "season_summary.md"
    render_report(args.template, context, summary_path)

    metadata_path = output_dir / "pipeline_metadata.json"
    metadata = {
        "target": args.target,
        "features": feature_names,
        "metrics": result.metrics,
        "cv_scores": result.cross_validation_scores,
        "artefacts": {
            "correlation_matrix": str(correlation_path),
            "residual_plot": str(residual_path),
            "feature_importance": str(feature_path),
            "summary": str(summary_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Analysis complete. Report saved to {summary_path}")


if __name__ == "__main__":
    main()
