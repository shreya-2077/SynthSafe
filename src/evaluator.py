import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wasserstein_distance


# ============================================================
# 1. COMPARE BASIC STATISTICS
# ============================================================

def compare_statistics(original, synthetic):

    # Select only numerical columns
    numerical_columns = original.select_dtypes(
        include=np.number
    ).columns

    results = []

    for column in numerical_columns:

        original_mean = original[column].mean()
        synthetic_mean = synthetic[column].mean()

        original_std = original[column].std()
        synthetic_std = synthetic[column].std()

        # Difference between means
        mean_difference = abs(
            original_mean - synthetic_mean
        )

        # Difference between standard deviations
        std_difference = abs(
            original_std - synthetic_std
        )

        # Distribution distance
        distance = wasserstein_distance(
            original[column],
            synthetic[column]
        )

        results.append({
            "column": column,

            "original_mean": round(
                original_mean, 2
            ),

            "synthetic_mean": round(
                synthetic_mean, 2
            ),

            "mean_difference": round(
                mean_difference, 2
            ),

            "original_std": round(
                original_std, 2
            ),

            "synthetic_std": round(
                synthetic_std, 2
            ),

            "std_difference": round(
                std_difference, 2
            ),

            "wasserstein_distance": round(
                distance, 2
            )
        })

    return pd.DataFrame(results)


# ============================================================
# 2. COMPARE CORRELATIONS
# ============================================================

def compare_correlations(original, synthetic):

    numerical_columns = original.select_dtypes(
        include=np.number
    ).columns

    original_corr = original[
        numerical_columns
    ].corr()

    synthetic_corr = synthetic[
        numerical_columns
    ].corr()

    # Absolute difference between correlation matrices
    difference = abs(
        original_corr - synthetic_corr
    )

    return (
        original_corr,
        synthetic_corr,
        difference
    )


# ============================================================
# 3. CREATE DISTRIBUTION PLOTS
# ============================================================

def save_distribution_plots(
    original,
    synthetic
):

    numerical_columns = original.select_dtypes(
        include=np.number
    ).columns

    for column in numerical_columns:

        plt.figure(figsize=(8, 5))

        sns.kdeplot(
            original[column],
            label="Original"
        )

        sns.kdeplot(
            synthetic[column],
            label="Synthetic"
        )

        plt.title(
            f"Distribution Comparison: {column}"
        )

        plt.xlabel(column)

        plt.ylabel("Density")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            f"reports/{column}_distribution.png"
        )

        plt.close()


# ============================================================
# 4. CREATE CORRELATION HEATMAPS
# ============================================================

def save_correlation_heatmaps(
    original,
    synthetic
):

    numerical_columns = original.select_dtypes(
        include=np.number
    ).columns

    original_corr = original[
        numerical_columns
    ].corr()

    synthetic_corr = synthetic[
        numerical_columns
    ].corr()

    # -------------------------
    # Original correlation
    # -------------------------

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        original_corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Original Dataset Correlation"
    )

    plt.tight_layout()

    plt.savefig(
        "reports/original_correlation.png"
    )

    plt.close()

    # -------------------------
    # Synthetic correlation
    # -------------------------

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        synthetic_corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Synthetic Dataset Correlation"
    )

    plt.tight_layout()

    plt.savefig(
        "reports/synthetic_correlation.png"
    )

    plt.close()


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # Make sure reports folder exists
    os.makedirs(
        "reports",
        exist_ok=True
    )

    # Load datasets
    original = pd.read_csv(
        "data/raw/customer_data.csv"
    )

    synthetic = pd.read_csv(
        "data/synthetic/synthetic_data.csv"
    )

    print("=" * 80)
    print("              SYNTHSAFE DATA EVALUATOR")
    print("=" * 80)

    print("\nOriginal dataset:")
    print(
        f"Rows: {len(original)}"
    )

    print(
        f"Columns: {len(original.columns)}"
    )

    print("\nSynthetic dataset:")
    print(
        f"Rows: {len(synthetic)}"
    )

    print(
        f"Columns: {len(synthetic.columns)}"
    )

    # ========================================================
    # Statistical comparison
    # ========================================================

    print("\n")
    print("=" * 80)
    print("STATISTICAL COMPARISON")
    print("=" * 80)

    statistics = compare_statistics(
        original,
        synthetic
    )

    print(
        statistics.to_string(
            index=False
        )
    )

    statistics.to_csv(
        "reports/statistical_comparison.csv",
        index=False
    )

    # ========================================================
    # Correlation comparison
    # ========================================================

    print("\n")
    print("=" * 80)
    print("CORRELATION DIFFERENCE")
    print("=" * 80)

    (
        original_corr,
        synthetic_corr,
        correlation_difference
    ) = compare_correlations(
        original,
        synthetic
    )

    print(
        correlation_difference.round(3)
    )

    correlation_difference.to_csv(
        "reports/correlation_difference.csv"
    )

    # ========================================================
    # Generate plots
    # ========================================================

    print("\nCreating distribution plots...")

    save_distribution_plots(
        original,
        synthetic
    )

    print("Distribution plots created.")

    print("\nCreating correlation heatmaps...")

    save_correlation_heatmaps(
        original,
        synthetic
    )

    print("Correlation heatmaps created.")

    # ========================================================
    # Finish
    # ========================================================

    print("\n")
    print("=" * 80)
    print("              EVALUATION COMPLETED")
    print("=" * 80)

    print("\nReports created inside:")
    print("reports/")