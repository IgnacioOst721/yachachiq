import csv
import numpy as np
from collections import defaultdict, Counter
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from landmark_utils import augment
import joblib

CSV_PATH = "asl_data.csv"
MODEL_PATH = "model.pkl"

# 1) Load the data you collected
X, y = [], []
with open(CSV_PATH) as f:
    r = csv.reader(f)
    next(r, None)  # skip header
    for row in r:
        if row:
            y.append(row[0])
            X.append([float(v) for v in row[1:]])
X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"Loaded {len(y)} samples across {len(set(y))} labels:")
for lab, n in sorted(Counter(y).items()):
    print(f"   {lab}: {n}")

if len(set(y)) < 2:
    print("\n[ERROR] Need at least 2 different labels to train.")
    raise SystemExit(1)

# 2) Add the geometry "hint" features
X = augment(X)

# 3) HONEST accuracy: hold out each label's LAST recording session as a test.
#    (A normal random split would look ~99% but that's misleading - the frames
#     are near-duplicates. This number is the real-world one.)
by = defaultdict(list)
for i, lab in enumerate(y):
    by[lab].append(i)
tr, te = [], []
for lab, idxs in by.items():
    k = 100 if len(idxs) > 100 else max(1, len(idxs) // 5)
    te += idxs[-k:]
    tr += idxs[:-k]
tr, te = np.array(tr), np.array(te)

check = ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1)
check.fit(X[tr], y[tr])
acc = accuracy_score(y[te], check.predict(X[te]))
print(f"\nHonest accuracy (on held-out signing): {acc*100:.1f}%")
print("(The live app's 'hold steady' filter makes the real-feel accuracy higher.)")

# 4) Train the FINAL model on ALL the data and save it
clf = ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1)
clf.fit(X, y)
joblib.dump(clf, MODEL_PATH)
print(f"\nSaved your model to {MODEL_PATH}  <-- THIS is your 'decoding file'")
