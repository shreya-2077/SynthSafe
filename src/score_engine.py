import os
import pandas as pd
import numpy as np


# ============================================================
# UTILITY SCORE
# ============================================================

def calculate_utility_score(statistics, correlation_difference):
    """
    Calculate a transparent utility score from 0-100.

    The score combines:
    1. Distribution similarity
    2. Correlation preservation

    This is a project-specific evaluation metric,
    not a universal industry standard.
    """

    # --------------------------------------------------------
    # Distribution similarity
    # --------------------------------------------------------

    distribution_errors = []

    for _, row in statistics.iterrows():

        original_mean = abs(row["original_mean"])
        mean_difference = abs(row["mean_difference"])

        original_std = abs(row["original_std"])
        std_difference = abs(row["std_difference"])

        # Avoid division by zero
        mean_error = (
            mean_difference / original_mean
            if original_mean > 1e-9
            else 0
        )

        std_error = (
            std_difference / original_std
            if original_std > 1e-9
            else 0
        )

        # Average the two relative errors
        distribution_error = (
            mean_error + std_error
        ) / 2

        distribution_errors.append(
            distribution_error
        )

    average_distribution_error = np.mean(
        distribution_errors
    )

    # Convert error into similarity
    distribution_similarity = (
        1 / (1 + average_distribution_error)
    )

    # --------------------------------------------------------
    # Correlation similarity
    # --------------------------------------------------------

    # Ignore diagonal zeros
    correlation_values = correlation_difference.values

    upper_triangle = correlation_values[
        np.triu_indices_from(
            correlation_values,
            k=1
        )
    ]

    if len(upper_triangle) > 0:

        average_correlation_difference = np.mean(
            np.abs(upper_triangle)
        )

    else:

        average_correlation_difference = 0

    # Convert correlation difference into similarity
    correlation_similarity = max(
        0,
        1 - average_correlation_difference
    )

    # --------------------------------------------------------
    # Combined utility
    # --------------------------------------------------------

    utility_score = (
        0.5 * distribution_similarity
        +
        0.5 * correlation_similarity
    ) * 100

    return {
        "utility_score": round(
            utility_score,
            2
        ),

        "distribution_similarity": round(
            distribution_similarity * 100,
            2
        ),

        "correlation_similarity": round(
            correlation_similarity * 100,
            2
        ),

        "average_distribution_error": round(
            average_distribution_error,
            4
        ),

        "average_correlation_difference": round(
            average_correlation_difference,
            4
        )
    }


# ============================================================
# PRIVACY INDICATORS
# ============================================================

def extract_privacy_indicators(privacy_report):

    def get_value(metric_name):

        result = privacy_report[
            privacy_report["metric"] == metric_name
        ]

        if len(result) == 0:
            return None

        return result.iloc[0]["value"]

    return {

        "exact_matches": get_value(
            "exact_matches"
        ),

        "duplicate_synthetic_records": get_value(
            "duplicate_synthetic_records"
        ),

        "minimum_distance": get_value(
            "minimum_synthetic_to_original_distance"
        ),

        "median_distance": get_value(
            "median_synthetic_to_original_distance"
        ),

        "average_distance": get_value(
            "average_synthetic_to_original_distance"
        ),

        "very_close_records": get_value(
            "very_close_synthetic_records"
        ),

        "very_close_percentage": get_value(
            "very_close_synthetic_percentage"
        )
    }


# ============================================================
# CREATE FINAL REPORT
# ============================================================

def create_summary():

    # --------------------------------------------------------
    # Load evaluator results
    # --------------------------------------------------------

    statistics = pd.read_csv(
        "reports/statistical_comparison.csv"
    )

    correlation_difference = pd.read_csv(
        "reports/correlation_difference.csv",
        index_col=0
    )

    privacy_report = pd.read_csv(
        "reports/privacy_report.csv"
    )

    # --------------------------------------------------------
    # Calculate utility
    # --------------------------------------------------------

    utility = calculate_utility_score(
        statistics,
        correlation_difference
    )

    # --------------------------------------------------------
    # Extract privacy indicators
    # --------------------------------------------------------

    privacy = extract_privacy_indicators(
        privacy_report
    )

    # --------------------------------------------------------
    # Create summary dataframe
    # --------------------------------------------------------

    summary = pd.DataFrame({

        "metric": [

            "Utility Score",
            "Distribution Similarity",
            "Correlation Similarity",
            "Average Distribution Error",
            "Average Correlation Difference",

            "Exact Matches",
            "Duplicate Synthetic Records",
            "Minimum Synthetic-to-Original Distance",
            "Median Synthetic-to-Original Distance",
            "Average Synthetic-to-Original Distance",
            "Very-Close Synthetic Records",
            "Very-Close Synthetic Percentage"
        ],

        "value": [

            utility["utility_score"],
            utility["distribution_similarity"],
            utility["correlation_similarity"],
            utility["average_distribution_error"],
            utility["average_correlation_difference"],

            privacy["exact_matches"],
            privacy["duplicate_synthetic_records"],
            privacy["minimum_distance"],
            privacy["median_distance"],
            privacy["average_distance"],
            privacy["very_close_records"],
            privacy["very_close_percentage"]
        ]
    })

    return summary, utility, privacy


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("                 SYNTHSAFE SCORE ENGINE")
    print("=" * 75)

    # Check required reports
    required_files = [

        "reports/statistical_comparison.csv",

        "reports/correlation_difference.csv",

        "reports/privacy_report.csv"
    ]

    for file in required_files:

        if not os.path.exists(file):

            raise FileNotFoundError(
                f"Required report not found: {file}\n"
                "Run the evaluator and privacy analyzer first."
            )

    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    summary, utility, privacy = create_summary()

    # --------------------------------------------------------
    # Display utility results
    # --------------------------------------------------------

    print("\n")
    print("-" * 75)
    print("UTILITY ANALYSIS")
    print("-" * 75)

    print(
        f"\nUtility Score: "
        f"{utility['utility_score']} / 100"
    )

    print(
        f"Distribution Similarity: "
        f"{utility['distribution_similarity']}%"
    )

    print(
        f"Correlation Similarity: "
        f"{utility['correlation_similarity']}%"
    )

    print(
        f"Average Distribution Error: "
        f"{utility['average_distribution_error']}"
    )

    print(
        f"Average Correlation Difference: "
        f"{utility['average_correlation_difference']}"
    )

    # --------------------------------------------------------
    # Display privacy indicators
    # --------------------------------------------------------

    print("\n")
    print("-" * 75)
    print("PRIVACY RISK INDICATORS")
    print("-" * 75)

    print(
        f"\nExact Matches: "
        f"{privacy['exact_matches']}"
    )

    print(
        f"Duplicate Synthetic Records: "
        f"{privacy['duplicate_synthetic_records']}"
    )

    print(
        f"Minimum Synthetic-to-Original Distance: "
        f"{privacy['minimum_distance']}"
    )

    print(
        f"Median Synthetic-to-Original Distance: "
        f"{privacy['median_distance']}"
    )

    print(
        f"Average Synthetic-to-Original Distance: "
        f"{privacy['average_distance']}"
    )

    print(
        f"Very-Close Synthetic Records: "
        f"{privacy['very_close_records']}"
    )

    print(
        f"Very-Close Percentage: "
        f"{privacy['very_close_percentage']}%"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary.to_csv(
        "reports/synthsafe_summary.csv",
        index=False
    )

    print("\n")
    print("=" * 75)
    print("              SYNTHSAFE ANALYSIS COMPLETED")
    print("=" * 75)

    print("\nSummary saved to:")
    print("reports/synthsafe_summary.csv")