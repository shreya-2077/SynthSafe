import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. CHECK EXACT RECORD MATCHES
# ============================================================

def check_exact_matches(original, synthetic):

    # Convert every value to string so comparisons
    # are consistent across data types
    original_copy = original.astype(str)
    synthetic_copy = synthetic.astype(str)

    # Find synthetic rows that are also present
    # in the original dataset
    merged = synthetic_copy.merge(
        original_copy.drop_duplicates(),
        how="inner"
    )

    exact_matches = len(merged)

    return exact_matches


# ============================================================
# 2. CHECK DUPLICATE SYNTHETIC RECORDS
# ============================================================

def check_duplicates(synthetic):

    duplicate_count = synthetic.duplicated().sum()

    return int(duplicate_count)


# ============================================================
# 3. PREPARE DATA FOR DISTANCE ANALYSIS
# ============================================================

def prepare_data(original, synthetic):

    # Numerical columns
    numerical_columns = original.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Categorical columns
    categorical_columns = original.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    transformers = []

    # Standardize numerical columns
    if numerical_columns:

        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numerical_columns
            )
        )

    # Convert categorical columns to one-hot vectors
    if categorical_columns:

        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # Fit using ORIGINAL data only
    preprocessor.fit(original)

    original_encoded = preprocessor.transform(
        original
    )

    synthetic_encoded = preprocessor.transform(
        synthetic
    )

    return (
        original_encoded,
        synthetic_encoded
    )


# ============================================================
# 4. NEAREST-NEIGHBOR PRIVACY ANALYSIS
# ============================================================

def nearest_neighbor_analysis(
    original_encoded,
    synthetic_encoded
):

    # --------------------------------------------------------
    # Synthetic -> Original
    # --------------------------------------------------------

    model = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean"
    )

    model.fit(original_encoded)

    synthetic_distances, _ = model.kneighbors(
        synthetic_encoded
    )

    synthetic_distances = (
        synthetic_distances.flatten()
    )

    # --------------------------------------------------------
    # Original -> Original
    #
    # We exclude each record's distance to itself.
    # This gives us a baseline for how close ordinary
    # real records are to one another.
    # --------------------------------------------------------

    if len(original_encoded) >= 2:

        real_model = NearestNeighbors(
            n_neighbors=2,
            metric="euclidean"
        )

        real_model.fit(original_encoded)

        real_distances, _ = real_model.kneighbors(
            original_encoded
        )

        # First neighbor is the record itself,
        # second neighbor is the closest OTHER record
        real_nearest_distances = (
            real_distances[:, 1]
        )

    else:

        real_nearest_distances = np.array([])

    return (
        synthetic_distances,
        real_nearest_distances
    )


# ============================================================
# 5. CREATE PRIVACY REPORT
# ============================================================

def create_privacy_report(
    original,
    synthetic
):

    print("\n")
    print("=" * 75)
    print("              SYNTHSAFE PRIVACY ANALYZER")
    print("=" * 75)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\nOriginal records:", len(original))
    print("Synthetic records:", len(synthetic))

    # --------------------------------------------------------
    # Exact matches
    # --------------------------------------------------------

    exact_matches = check_exact_matches(
        original,
        synthetic
    )

    print("\nExact synthetic/original matches:")
    print(exact_matches)

    # --------------------------------------------------------
    # Duplicate synthetic records
    # --------------------------------------------------------

    duplicate_count = check_duplicates(
        synthetic
    )

    print("\nDuplicate synthetic records:")
    print(duplicate_count)

    # --------------------------------------------------------
    # Distance analysis
    # --------------------------------------------------------

    (
        original_encoded,
        synthetic_encoded
    ) = prepare_data(
        original,
        synthetic
    )

    (
        synthetic_distances,
        real_nearest_distances
    ) = nearest_neighbor_analysis(
        original_encoded,
        synthetic_encoded
    )

    # --------------------------------------------------------
    # Synthetic -> Original statistics
    # --------------------------------------------------------

    minimum_distance = np.min(
        synthetic_distances
    )

    average_distance = np.mean(
        synthetic_distances
    )

    median_distance = np.median(
        synthetic_distances
    )

    percentile_5 = np.percentile(
        synthetic_distances,
        5
    )

    percentile_25 = np.percentile(
        synthetic_distances,
        25
    )

    # --------------------------------------------------------
    # Real -> Real baseline
    # --------------------------------------------------------

    if len(real_nearest_distances) > 0:

        real_median_distance = np.median(
            real_nearest_distances
        )

        real_5th_percentile = np.percentile(
            real_nearest_distances,
            5
        )

    else:

        real_median_distance = np.nan
        real_5th_percentile = np.nan

    # --------------------------------------------------------
    # Compare synthetic proximity with real-data baseline
    # --------------------------------------------------------

    if (
        not np.isnan(real_5th_percentile)
        and real_5th_percentile > 0
    ):

        very_close_count = np.sum(
            synthetic_distances
            <= real_5th_percentile
        )

        very_close_percentage = (
            very_close_count
            / len(synthetic_distances)
            * 100
        )

    else:

        very_close_count = 0
        very_close_percentage = 0

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n")
    print("-" * 75)
    print("NEAREST-NEIGHBOR ANALYSIS")
    print("-" * 75)

    print(
        "\nMinimum synthetic -> original distance:",
        round(minimum_distance, 4)
    )

    print(
        "Average synthetic -> original distance:",
        round(average_distance, 4)
    )

    print(
        "Median synthetic -> original distance:",
        round(median_distance, 4)
    )

    print(
        "5th percentile distance:",
        round(percentile_5, 4)
    )

    print(
        "25th percentile distance:",
        round(percentile_25, 4)
    )

    print(
        "\nMedian real -> nearest-real distance:",
        round(real_median_distance, 4)
    )

    print(
        "5th percentile real -> nearest-real distance:",
        round(real_5th_percentile, 4)
    )

    print(
        "\nSynthetic records at or below the",
        "5th-percentile real-record distance:",
        very_close_count
    )

    print(
        "Percentage of synthetic records in this",
        "very-close region:",
        round(very_close_percentage, 2),
        "%"
    )

    # --------------------------------------------------------
    # Create report dataframe
    # --------------------------------------------------------

    report = pd.DataFrame({
        "metric": [
            "original_records",
            "synthetic_records",
            "exact_matches",
            "duplicate_synthetic_records",
            "minimum_synthetic_to_original_distance",
            "average_synthetic_to_original_distance",
            "median_synthetic_to_original_distance",
            "synthetic_5th_percentile_distance",
            "synthetic_25th_percentile_distance",
            "median_real_to_nearest_real_distance",
            "real_5th_percentile_distance",
            "very_close_synthetic_records",
            "very_close_synthetic_percentage"
        ],

        "value": [
            len(original),
            len(synthetic),
            exact_matches,
            duplicate_count,
            round(minimum_distance, 6),
            round(average_distance, 6),
            round(median_distance, 6),
            round(percentile_5, 6),
            round(percentile_25, 6),
            round(real_median_distance, 6),
            round(real_5th_percentile, 6),
            int(very_close_count),
            round(very_close_percentage, 4)
        ]
    })

    return report


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # Make sure reports directory exists
    os.makedirs(
        "reports",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    original = pd.read_csv(
        "data/raw/customer_data.csv"
    )

    synthetic = pd.read_csv(
        "data/synthetic/synthetic_data.csv"
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    if list(original.columns) != list(
        synthetic.columns
    ):

        raise ValueError(
            "Original and synthetic datasets "
            "must have the same columns."
        )

    # --------------------------------------------------------
    # Create privacy report
    # --------------------------------------------------------

    report = create_privacy_report(
        original,
        synthetic
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = (
        "reports/privacy_report.csv"
    )

    report.to_csv(
        report_path,
        index=False
    )

    print("\n")
    print("=" * 75)
    print("PRIVACY ANALYSIS COMPLETED")
    print("=" * 75)

    print("\nReport saved to:")
    print(report_path)