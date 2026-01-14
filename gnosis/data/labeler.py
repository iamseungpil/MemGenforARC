"""
Correctness labeler for completion data.

Processes shard files and adds correctness labels using task-specific evaluators.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from gnosis.data.evaluator import get_evaluator, BaseEvaluator

logger = logging.getLogger(__name__)


class Labeler:
    """
    Labels completion data with correctness based on task type.

    Usage:
        labeler = Labeler(task="math")

        # Label single shard
        labeled_df = labeler.label_shard("shard-0000.parquet")

        # Label entire directory
        labeler.label_directory(
            input_dir="./raw_generations",
            output_dir="./verified"
        )
    """

    def __init__(
        self,
        task: str,
        completion_col: str = "completion",
        ground_truth_col: str = "ground_truth",
        question_col: str = "question",
    ):
        """
        Initialize the labeler.

        Args:
            task: Task type - "math", "trivia", or "mcq".
            completion_col: Column name for model completions.
            ground_truth_col: Column name for ground truth answers.
            question_col: Column name for questions.
        """
        self.task = task
        self.evaluator: type = get_evaluator(task)
        self.completion_col = completion_col
        self.ground_truth_col = ground_truth_col
        self.question_col = question_col

        logger.info(f"Initialized Labeler for task: {task}")

    def label_shard(
        self,
        shard_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Label a single shard file.

        Args:
            shard_path: Path to input parquet/csv file.
            output_path: Optional output path (auto-generated if not provided).

        Returns:
            DataFrame with added correctness_label and pred_parsed columns.
        """
        shard_path = Path(shard_path)

        # Load data
        if shard_path.suffix == ".parquet":
            df = pd.read_parquet(shard_path)
        elif shard_path.suffix == ".csv":
            df = pd.read_csv(shard_path)
        else:
            raise ValueError(f"Unsupported file format: {shard_path.suffix}")

        logger.info(f"Labeling shard: {shard_path} ({len(df)} rows)")

        # Validate columns
        required_cols = [self.completion_col, self.ground_truth_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Label each row
        correctness_labels = []
        pred_parsed = []

        for idx, row in df.iterrows():
            completion = str(row[self.completion_col])
            ground_truth = str(row[self.ground_truth_col])

            is_correct, is_parsed = self.evaluator.evaluate(completion, ground_truth)

            correctness_labels.append(int(is_correct))
            pred_parsed.append(is_parsed)

        # Add columns
        df["correctness_label"] = correctness_labels
        df["pred_parsed"] = pred_parsed
        df["task"] = self.task

        # Compute statistics
        num_correct = sum(correctness_labels)
        num_parsed = sum(pred_parsed)
        total = len(df)

        logger.info(
            f"Shard stats: {num_correct}/{total} correct "
            f"({100*num_correct/total:.1f}%), "
            f"{num_parsed}/{total} parsed "
            f"({100*num_parsed/total:.1f}%)"
        )

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == ".parquet":
                df.to_parquet(output_path, index=False)
            else:
                df.to_csv(output_path, index=False)

            logger.info(f"Saved labeled shard to: {output_path}")

        return df

    def label_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "shard-*.parquet",
    ) -> pd.DataFrame:
        """
        Label all shard files in a directory.

        Args:
            input_dir: Directory containing input shard files.
            output_dir: Directory to save labeled outputs.
            pattern: Glob pattern for shard files.

        Returns:
            Combined DataFrame of all labeled shards.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find shard files
        shard_files = sorted(input_dir.glob(pattern))
        if not shard_files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {input_dir}"
            )

        logger.info(f"Found {len(shard_files)} shard files to label")

        all_dfs = []
        total_correct = 0
        total_parsed = 0
        total_rows = 0

        for shard_path in shard_files:
            # Generate output filename
            output_name = shard_path.stem + ".verified" + shard_path.suffix
            output_path = output_dir / output_name

            # Label shard
            labeled_df = self.label_shard(shard_path, output_path)
            all_dfs.append(labeled_df)

            # Accumulate stats
            total_correct += labeled_df["correctness_label"].sum()
            total_parsed += labeled_df["pred_parsed"].sum()
            total_rows += len(labeled_df)

        # Log overall statistics
        logger.info(
            f"Overall stats: {total_correct}/{total_rows} correct "
            f"({100*total_correct/total_rows:.1f}%), "
            f"{total_parsed}/{total_rows} parsed "
            f"({100*total_parsed/total_rows:.1f}%)"
        )

        return pd.concat(all_dfs, ignore_index=True)

    @staticmethod
    def compute_statistics(df: pd.DataFrame) -> dict:
        """
        Compute statistics for a labeled DataFrame.

        Args:
            df: DataFrame with correctness_label and pred_parsed columns.

        Returns:
            Dictionary of statistics.
        """
        total = len(df)
        if total == 0:
            return {
                "total": 0,
                "correct": 0,
                "parsed": 0,
                "accuracy": 0.0,
                "parse_rate": 0.0,
            }

        correct = df["correctness_label"].sum()
        parsed = df["pred_parsed"].sum()

        return {
            "total": total,
            "correct": int(correct),
            "parsed": int(parsed),
            "accuracy": correct / total,
            "parse_rate": parsed / total,
            "accuracy_given_parsed": correct / parsed if parsed > 0 else 0.0,
        }

    @staticmethod
    def compute_per_question_stats(
        df: pd.DataFrame,
        question_col: str = "question",
    ) -> pd.DataFrame:
        """
        Compute per-question statistics.

        Useful for analyzing questions with multiple completions.

        Args:
            df: Labeled DataFrame.
            question_col: Column name for questions.

        Returns:
            DataFrame with per-question statistics.
        """
        grouped = df.groupby(question_col).agg({
            "correctness_label": ["sum", "count"],
            "pred_parsed": "sum",
        }).reset_index()

        grouped.columns = [
            question_col,
            "num_correct",
            "num_completions",
            "num_parsed",
        ]

        grouped["accuracy"] = grouped["num_correct"] / grouped["num_completions"]
        grouped["parse_rate"] = grouped["num_parsed"] / grouped["num_completions"]

        # Classify questions
        def classify(row):
            if row["num_correct"] == row["num_completions"]:
                return "all_correct"
            elif row["num_correct"] == 0:
                return "all_wrong"
            else:
                return "mixed"

        grouped["category"] = grouped.apply(classify, axis=1)

        return grouped
