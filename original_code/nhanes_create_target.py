import pandas as pd

# =========================
# Load dataset
# =========================
df = pd.read_csv(
    r"C:\Users\<user>\Research center\Heartseanse2.0\Dataset2\nhanes_cleaned_step1.csv"
)

# =========================
# Target columns
# =========================
target_cols = [
    "MCQ160B",  # Congestive heart failure
    "MCQ160C",  # Coronary heart disease
    "MCQ160D",  # Angina
    "MCQ160E"   # Heart attack
]

# =========================
# Remove invalid code 9
# =========================
for col in target_cols:
    df[col] = df[col].replace(9, pd.NA)

# =========================
# Create binary target
# =========================

def create_target(row):

    vals = row[target_cols]

    # Positive if ANY = 1
    if (vals == 1).any():
        return 1

    # Negative if ALL valid answers are 2
    if (vals == 2).all():
        return 0

    # Otherwise unresolved
    return pd.NA

df["CVD_TARGET"] = df.apply(create_target, axis=1)

# =========================
# Remove unresolved rows
# =========================
df = df.dropna(subset=["CVD_TARGET"])

# Convert target to integer
df["CVD_TARGET"] = df["CVD_TARGET"].astype(int)

# =========================
# Print class distribution
# =========================
print("\nFinal Dataset Shape:", df.shape)

print("\nTarget Distribution:\n")
print(df["CVD_TARGET"].value_counts())

print("\nTarget Percentage:\n")
print(df["CVD_TARGET"].value_counts(normalize=True) * 100)

# =========================
# Save dataset
# =========================
output_path = r"C:\Users\<user>\Research center\Heartseanse2.0\Dataset2\nhanes_final_with_target.csv"

df.to_csv(output_path, index=False)

print("\nSaved:", output_path)