import numpy as np
import pandas as pd

from scipy.stats import norm


# ============================================================
# GAUSSIAN COPULA SYNTHETIC DATA GENERATOR
# ============================================================

def generate_synthetic_data(
    df,
    random_state=42
):
    """
    Generate synthetic data while attempting to preserve:

    1. Original numerical distributions
    2. Relationships between numerical variables
    3. Original categorical frequencies

    This is a baseline Gaussian-copula approach.
    """

    rng = np.random.default_rng(random_state)

    synthetic = pd.DataFrame(
        index=range(len(df))
    )

    # --------------------------------------------------------
    # Separate numerical and categorical columns
    # --------------------------------------------------------

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    # ========================================================
    # NUMERICAL DATA
    # ========================================================

    if len(numerical_columns) > 0:

        numerical_data = df[
            numerical_columns
        ].copy()

        # ----------------------------------------------------
        # Step 1: Convert each numerical column into
        # approximately standard normal values using ranks
        # ----------------------------------------------------

        gaussian_data = pd.DataFrame(
            index=numerical_data.index
        )

        for column in numerical_columns:

            values = numerical_data[column]

            # Rank the observations
            ranks = values.rank(
                method="average"
            )

            # Convert ranks into probabilities
            probabilities = (
                ranks - 0.5
            ) / len(values)

            # Convert probabilities to normal scores
            gaussian_values = norm.ppf(
                probabilities
            )

            gaussian_data[column] = gaussian_values

        # ----------------------------------------------------
        # Step 2: Learn the dependency/correlation structure
        # ----------------------------------------------------

        correlation_matrix = gaussian_data.corr().values

        # Numerical stability
        correlation_matrix = (
            correlation_matrix
            + correlation_matrix.T
        ) / 2

        # Add tiny value to diagonal
        correlation_matrix += (
            np.eye(
                len(numerical_columns)
            ) * 1e-6
        )

        # ----------------------------------------------------
        # Step 3: Cholesky decomposition
        # ----------------------------------------------------

        try:

            L = np.linalg.cholesky(
                correlation_matrix
            )

        except np.linalg.LinAlgError:

            # Fallback for numerical stability
            eigenvalues, eigenvectors = np.linalg.eigh(
                correlation_matrix
            )

            eigenvalues = np.maximum(
                eigenvalues,
                1e-6
            )

            correlation_matrix = (
                eigenvectors
                @ np.diag(eigenvalues)
                @ eigenvectors.T
            )

            L = np.linalg.cholesky(
                correlation_matrix
            )

        # ----------------------------------------------------
        # Step 4: Generate independent random normal values
        # ----------------------------------------------------

        independent_random = rng.normal(
            0,
            1,
            size=(
                len(df),
                len(numerical_columns)
            )
        )

        # ----------------------------------------------------
        # Step 5: Introduce learned correlations
        # ----------------------------------------------------

        correlated_random = (
            independent_random @ L.T
        )

        # ----------------------------------------------------
        # Step 6: Convert generated values into
        # probabilities
        # ----------------------------------------------------

        generated_probabilities = norm.cdf(
            correlated_random
        )

        # ----------------------------------------------------
        # Step 7: Map probabilities back to the
        # original empirical distributions
        # ----------------------------------------------------

        for i, column in enumerate(
            numerical_columns
        ):

            original_values = (
                numerical_data[column]
                .sort_values()
                .to_numpy()
            )

            probabilities = (
                generated_probabilities[:, i]
            )

            # Convert probabilities into
            # positions inside original distribution
            positions = (
                probabilities
                * (len(original_values) - 1)
            )

            lower_indices = np.floor(
                positions
            ).astype(int)

            upper_indices = np.ceil(
                positions
            ).astype(int)

            fraction = (
                positions
                - lower_indices
            )

            generated_values = (
                original_values[lower_indices]
                * (1 - fraction)
                +
                original_values[upper_indices]
                * fraction
            )

            # Preserve integer columns
            if pd.api.types.is_integer_dtype(
                numerical_data[column]
            ):

                generated_values = np.round(
                    generated_values
                ).astype(int)

            # Keep values inside original range
            generated_values = np.clip(
                generated_values,
                numerical_data[column].min(),
                numerical_data[column].max()
            )

            synthetic[column] = generated_values

    # ========================================================
    # CATEGORICAL DATA
    # ========================================================

    for column in categorical_columns:

        probabilities = (
            df[column]
            .value_counts(
                normalize=True
            )
        )

        synthetic[column] = rng.choice(
            probabilities.index,
            size=len(df),
            p=probabilities.values
        )

    # --------------------------------------------------------
    # Keep columns in the original order
    # --------------------------------------------------------

    synthetic = synthetic[
        df.columns
    ]

    return synthetic


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("          SYNTHSAFE V1 SYNTHETIC DATA GENERATOR")
    print("=" * 75)

    # --------------------------------------------------------
    # Load original dataset
    # --------------------------------------------------------

    original = pd.read_csv(
        "data/raw/customer_data.csv"
    )

    print("\nOriginal dataset:")
    print(
        f"Rows: {len(original)}"
    )

    print(
        f"Columns: {len(original.columns)}"
    )

    # --------------------------------------------------------
    # Generate synthetic dataset
    # --------------------------------------------------------

    synthetic = generate_synthetic_data(
        original,
        random_state=42
    )

    # --------------------------------------------------------
    # Save synthetic dataset
    # --------------------------------------------------------

    synthetic.to_csv(
        "data/synthetic/synthetic_data.csv",
        index=False
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nSynthetic dataset generated successfully!")

    print(
        f"Rows: {len(synthetic)}"
    )

    print(
        f"Columns: {len(synthetic.columns)}"
    )

    print("\nFirst 5 synthetic records:")

    print(
        synthetic.head()
    )

    print("\nSaved to:")

    print(
        "data/synthetic/synthetic_data.csv"
    )

    print("\nV1 generation completed successfully!")