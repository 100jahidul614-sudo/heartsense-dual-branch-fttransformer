import pandas as pd

# =========================
# Load dataset
# =========================
df = pd.read_csv(
    r"C:\Users\<user>\Research center\Heartseanse2.0\Dataset2\nhanes_2017_2020_merged_raw.csv"
)

print("Original Shape:", df.shape)

# =========================
# Remove features with >70% missing
# =========================
missing_percent = df.isnull().mean() * 100

cols_to_keep = missing_percent[missing_percent <= 70].index

df = df[cols_to_keep]

print("After Removing >70% Missing Columns:", df.shape)

# =========================
# Save cleaned version
# =========================
output_path = r"C:\Users\<user>\Research center\Heartseanse2.0\Dataset2\nhanes_cleaned_step1.csv"

df.to_csv(output_path, index=False)

print("Saved:", output_path)