"""
Dataset rebalancer for training data preparation.

Balances data across correctness categories (all_correct, all_wrong, mixed)
using downsampling or upsampling strategies.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union, Literal

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


StrategyType = Literal["downsample", "upsample", "none"]


class Rebalancer:
    """
    Rebalances labeled completion data for training.

    Classifies questions into buckets based on completion correctness:
    - all_correct: All completions for a question are correct
    - all_wrong: All completions for a question are wrong
    - mixed: Some completions correct, some wrong

    Usage:
        rebalancer = Rebalancer(strategy="downsample")
        rebalancer.rebalance(
            input_dirs=["./model_a/verified", "./model_b/verified"],
            output_dir="./Final"
        )
    """

    def __init__(
        self,
        strategy: StrategyType = "downsample",
        question_col: str = "question",
        correctness_col: str = "correctness_label",
        random_seed: int = 42,
    ):
        """
        Initialize the rebalancer.

        Args:
            strategy: Balancing strategy - "downsample", "upsample", or "none".
            question_col: Column name for questions.
            correctness_col: Column name for correctness labels.
            random_seed: Random seed for reproducibility.
        """
        self.strategy = strategy
        self.question_col = question_col
        self.correctness_col = correctness_col
        self.random_seed = random_seed

        logger.info(f"Initialized Rebalancer with strategy: {strategy}")

    def _classify_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify questions into buckets based on completion correctness.

        Args:
            df: DataFrame with completion data.

        Returns:
            DataFrame with question-level statistics and bucket classification.
        """
        # Group by question and compute stats
        grouped = df.groupby(self.question_col).agg({
            self.correctness_col: ["sum", "count"],
        }).reset_index()

        grouped.columns = [self.question_col, "num_correct", "num_completions"]

        # Classify into buckets
        def classify(row):
            if row["num_correct"] == row["num_completions"]:
                return "all_correct"
            elif row["num_correct"] == 0:
                return "all_wrong"
            else:
                return "mixed"

        grouped["bucket"] = grouped.apply(classify, axis=1)

        return grouped

    def _balance_buckets(
        self,
        question_stats: pd.DataFrame,
        target_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Balance question buckets according to strategy.

        Args:
            question_stats: DataFrame with question classifications.
            target_size: Target size per bucket (auto-determined if None).

        Returns:
            DataFrame with balanced question set.
        """
        bucket_counts = question_stats["bucket"].value_counts()
        logger.info(f"Bucket counts before balancing: {bucket_counts.to_dict()}")

        if self.strategy == "none":
            return question_stats

        # Determine target size
        if target_size is None:
            if self.strategy == "downsample":
                target_size = bucket_counts.min()
            elif self.strategy == "upsample":
                target_size = bucket_counts.max()
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        logger.info(f"Target size per bucket: {target_size}")

        # Balance each bucket
        np.random.seed(self.random_seed)
        balanced_dfs = []

        for bucket_name in ["all_correct", "all_wrong", "mixed"]:
            bucket_df = question_stats[question_stats["bucket"] == bucket_name]
            current_size = len(bucket_df)

            if current_size == 0:
                logger.warning(f"Empty bucket: {bucket_name}")
                continue

            if current_size == target_size:
                balanced_dfs.append(bucket_df)
            elif current_size > target_size:
                # Downsample
                sampled = bucket_df.sample(
                    n=target_size,
                    random_state=self.random_seed,
                    replace=False,
                )
                balanced_dfs.append(sampled)
            else:
                # Upsample
                if self.strategy == "upsample":
                    # Sample with replacement to reach target
                    sampled = bucket_df.sample(
                        n=target_size,
                        random_state=self.random_seed,
                        replace=True,
                    )
                    balanced_dfs.append(sampled)
                else:
                    # For downsample strategy, keep original
                    balanced_dfs.append(bucket_df)

        if not balanced_dfs:
            raise ValueError("No valid buckets found")

        balanced = pd.concat(balanced_dfs, ignore_index=True)

        bucket_counts_after = balanced["bucket"].value_counts()
        logger.info(f"Bucket counts after balancing: {bucket_counts_after.to_dict()}")

        return balanced

    def rebalance(
        self,
        input_dirs: List[Union[str, Path]],
        output_dir: Union[str, Path],
        pattern: str = "*.verified.parquet",
        output_filename: str = "merged_balanced.parquet",
        target_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Rebalance data from multiple input directories.

        Args:
            input_dirs: List of directories containing verified shard files.
            output_dir: Output directory for balanced data.
            pattern: Glob pattern for input files.
            output_filename: Name of output file.
            target_size: Target samples per bucket (auto if None).

        Returns:
            Balanced DataFrame.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load all data
        all_dfs = []
        for input_dir in input_dirs:
            input_dir = Path(input_dir)
            files = sorted(input_dir.glob(pattern))

            if not files:
                logger.warning(f"No files matching '{pattern}' in {input_dir}")
                continue

            logger.info(f"Loading {len(files)} files from {input_dir}")

            for f in files:
                df = pd.read_parquet(f)
                all_dfs.append(df)

        if not all_dfs:
            raise FileNotFoundError("No input files found")

        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined data: {len(combined_df)} rows")

        # Classify questions
        question_stats = self._classify_questions(combined_df)
        logger.info(f"Total unique questions: {len(question_stats)}")

        # Balance
        balanced_questions = self._balance_buckets(question_stats, target_size)

        # Filter original data to keep only balanced questions
        balanced_question_set = set(balanced_questions[self.question_col])

        # Handle potential duplicates from upsampling
        if self.strategy == "upsample":
            # Need to duplicate rows for upsampled questions
            question_counts = balanced_questions[self.question_col].value_counts()
            result_dfs = []

            for question, count in question_counts.items():
                question_data = combined_df[
                    combined_df[self.question_col] == question
                ]
                # Repeat the data 'count' times
                for _ in range(count):
                    result_dfs.append(question_data)

            balanced_df = pd.concat(result_dfs, ignore_index=True)
        else:
            balanced_df = combined_df[
                combined_df[self.question_col].isin(balanced_question_set)
            ].copy()

        # Shuffle
        balanced_df = balanced_df.sample(
            frac=1.0,
            random_state=self.random_seed,
        ).reset_index(drop=True)

        logger.info(f"Final balanced data: {len(balanced_df)} rows")

        # Save
        output_path = output_dir / output_filename
        balanced_df.to_parquet(output_path, index=False)
        logger.info(f"Saved balanced data to: {output_path}")

        # Also save statistics
        stats = self.compute_balance_statistics(balanced_df)
        stats_path = output_dir / "balance_statistics.txt"
        with open(stats_path, "w") as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Saved statistics to: {stats_path}")

        return balanced_df

    def compute_balance_statistics(self, df: pd.DataFrame) -> dict:
        """
        Compute statistics for balanced data.

        Args:
            df: Balanced DataFrame.

        Returns:
            Dictionary of statistics.
        """
        total_rows = len(df)
        unique_questions = df[self.question_col].nunique()

        # Re-classify to get bucket distribution
        question_stats = self._classify_questions(df)
        bucket_counts = question_stats["bucket"].value_counts()

        stats = {
            "total_rows": total_rows,
            "unique_questions": unique_questions,
            "avg_completions_per_question": total_rows / unique_questions if unique_questions > 0 else 0,
            "bucket_all_correct": bucket_counts.get("all_correct", 0),
            "bucket_all_wrong": bucket_counts.get("all_wrong", 0),
            "bucket_mixed": bucket_counts.get("mixed", 0),
            "overall_accuracy": df[self.correctness_col].mean(),
            "strategy": self.strategy,
        }

        return stats

    @staticmethod
    def analyze_distribution(
        df: pd.DataFrame,
        question_col: str = "question",
        correctness_col: str = "correctness_label",
    ) -> dict:
        """
        Analyze the distribution of a dataset.

        Args:
            df: DataFrame to analyze.
            question_col: Question column name.
            correctness_col: Correctness column name.

        Returns:
            Dictionary with distribution analysis.
        """
        # Question-level stats
        grouped = df.groupby(question_col).agg({
            correctness_col: ["sum", "count"],
        }).reset_index()
        grouped.columns = [question_col, "num_correct", "num_completions"]

        def classify(row):
            if row["num_correct"] == row["num_completions"]:
                return "all_correct"
            elif row["num_correct"] == 0:
                return "all_wrong"
            else:
                return "mixed"

        grouped["bucket"] = grouped.apply(classify, axis=1)

        bucket_counts = grouped["bucket"].value_counts().to_dict()

        return {
            "total_rows": len(df),
            "unique_questions": len(grouped),
            "completions_per_question": grouped["num_completions"].mean(),
            "bucket_distribution": bucket_counts,
            "bucket_percentages": {
                k: v / len(grouped) * 100
                for k, v in bucket_counts.items()
            },
            "overall_accuracy": df[correctness_col].mean(),
        }
