import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000

age = np.random.randint(18, 65, N)

experience = np.maximum(
    0,
    age - np.random.randint(18, 30, N)
)

education = np.random.choice(
    ["Bachelor", "Master", "PhD"],
    N,
    p=[0.55, 0.35, 0.10]
)

city = np.random.choice(
    ["Delhi", "Mumbai", "Bangalore", "Pune", "Jaipur"],
    N,
    p=[0.20, 0.20, 0.25, 0.20, 0.15]
)

salary = (
    25000
    + experience * 6500
    + age * 500
    + np.random.normal(0, 15000, N)
)

salary = np.maximum(
    salary,
    18000
).astype(int)

spending = (
    salary * np.random.uniform(0.25, 0.60, N)
    + np.random.normal(0, 5000, N)
)

spending = np.maximum(
    spending,
    5000
).astype(int)

credit_score = (
    550
    + experience * 5
    + np.random.normal(0, 40, N)
)

credit_score = np.clip(
    credit_score,
    300,
    850
).astype(int)

df = pd.DataFrame({
    "age": age,
    "experience": experience,
    "education": education,
    "city": city,
    "salary": salary,
    "monthly_spending": spending,
    "credit_score": credit_score
})

df.to_csv(
    "data/raw/customer_data.csv",
    index=False
)

print("Dataset created successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nFirst 5 rows:")
print(df.head())