import numpy as np
import pandas as pd


def generate_synthetic_data(df, random_state=42):

    # Make results reproducible
    np.random.seed(random_state)

    synthetic = pd.DataFrame()

    # Process every column
    for column in df.columns:

        series = df[column]

        # ==========================================
        # NUMERICAL COLUMNS
        # ==========================================

        if pd.api.types.is_numeric_dtype(series):

            mean = series.mean()
            std = series.std()

            # Generate values using a normal distribution
            values = np.random.normal(
                loc=mean,
                scale=std,
                size=len(df)
            )

            # Keep generated values within
            # the original minimum and maximum
            values = np.clip(
                values,
                series.min(),
                series.max()
            )

            # Keep integer columns as integers
            if pd.api.types.is_integer_dtype(series):

                values = np.round(values).astype(int)

            synthetic[column] = values

        # ==========================================
        # CATEGORICAL COLUMNS
        # ==========================================

        else:

            # Calculate probability of each category
            probabilities = series.value_counts(
                normalize=True
            )

            # Generate new categories using
            # the same approximate probabilities
            synthetic[column] = np.random.choice(
                probabilities.index,
                size=len(df),
                p=probabilities.values
            )

    return synthetic


# ==============================================
# RUN GENERATOR
# ==============================================

if __name__ == "__main__":

    # Load original dataset
    original = pd.read_csv(
        "data/raw/customer_data.csv"
    )

    print("=" * 70)
    print("             SYNTHSAFE SYNTHETIC GENERATOR")
    print("=" * 70)

    print("\nOriginal dataset:")
    print(f"Rows: {len(original)}")
    print(f"Columns: {len(original.columns)}")

    # Generate synthetic data
    synthetic = generate_synthetic_data(
        original
    )

    # Save synthetic dataset
    synthetic.to_csv(
        "data/synthetic/synthetic_data.csv",
        index=False
    )

    print("\nSynthetic dataset generated successfully!")

    print("\nSynthetic dataset:")
    print(f"Rows: {len(synthetic)}")
    print(f"Columns: {len(synthetic.columns)}")

    print("\nFirst 5 synthetic records:")
    print(synthetic.head())

    print("\nSaved to:")
    print("data/synthetic/synthetic_data.csv")