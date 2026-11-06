import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Load OOF predictions
lgb = pd.read_csv(r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\phase2_allmodels_allfeatures\LightGBM_oof.csv")
rf = pd.read_csv(r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\phase2_allmodels_allfeatures\RF_oof.csv")
cnn = pd.read_csv(r"C:\Users\<user>\Research center\HeartSense_repro\runs\brfss\phase2_allmodels_allfeatures\CNN1D_oof.csv")

# Make sure same order
assert all(lgb['y_true'] == rf['y_true'])
assert all(lgb['y_true'] == cnn['y_true'])

y_true = lgb['y_true'].values

# Simple average fusion
p_avg = (lgb['y_prob'] + rf['y_prob'] + cnn['y_prob']) / 3

# Threshold = 0.50
y_pred = (p_avg >= 0.5).astype(int)

# Metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("FP:", fp)
print("FN:", fn)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)