import pandas as pd


def profile_dataset(file_path):

    # Load the dataset
    df = pd.read_csv(file_path)

    profile = []

    # Analyze every column
    for column in df.columns:

        series = df[column]

        # Basic information
        information = {
            "column": column,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique())
        }

        # Numerical columns
        if pd.api.types.is_numeric_dtype(series):

            information.update({
                "mean": round(series.mean(), 2),
                "median": round(series.median(), 2),
                "std": round(series.std(), 2),
                "min": round(series.min(), 2),
                "max": round(series.max(), 2)
            })

        # Categorical columns
        else:

            information.update({
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None
            })

        profile.append(information)

    return pd.DataFrame(profile)


# Run the profiler directly
if __name__ == "__main__":

    file_path = "data/raw/customer_data.csv"

    report = profile_dataset(file_path)

    print("\n")
    print("=" * 80)
    print("                 SYNTHSAFE DATA PROFILE")
    print("=" * 80)

    print(report.to_string(index=False))

    print("\n")
    print("Profiling completed successfully!")