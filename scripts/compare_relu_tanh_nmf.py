
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import NMF


# =====================================
# 1. Configuration
# =====================================

ROOT = Path(__file__).resolve().parents[1]

ACTIVATION_DIR = ROOT / "reports" / "activations"
OUTPUT_DIR = ROOT / "reports" / "nmf"
METRICS_DIR = ROOT / "reports" / "metrics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

ranks = [4, 8, 12, 16, 24, 32, 48, 64]


# =====================================
# 2. Load Tanh activations
# =====================================

activation_path = ACTIVATION_DIR / "tanh_activations.npz"

with np.load(activation_path) as data:
    A = data["activations"].astype(np.float64)
    labels = data["labels"]

# Keep only positive activations
A_positive = np.maximum(A, 0)

print("===== Tanh Positive Activations =====")
print("Shape:", A_positive.shape)
print("Minimum:", A_positive.min())
print("Maximum:", A_positive.max())

assert np.all(A_positive >= 0)


# =====================================
# 3. NMF decomposition
# =====================================

results = []

for k in ranks:

    print(f"\n===== Running NMF: k={k} =====")

    model = NMF(
        n_components=k,
        init="nndsvda",
        solver="cd",
        max_iter=500,
        random_state=42
    )

    start_time = time.time()

    W = model.fit_transform(A_positive)
    H = model.components_

    elapsed = time.time() - start_time

    relative_error = (
        np.linalg.norm(A_positive - W @ H, ord="fro")
        / np.linalg.norm(A_positive, ord="fro")
    )

    result = {
        "rank": k,
        "reconstruction_error": float(relative_error),
        "iterations": int(model.n_iter_),
        "time_seconds": float(elapsed)
    }

    results.append(result)

    print("W shape:", W.shape)
    print("H shape:", H.shape)
    print("Relative error:", relative_error)
    print("Iterations:", model.n_iter_)
    print("Time:", round(elapsed, 2), "seconds")

    np.savez_compressed(
        OUTPUT_DIR / f"tanh_positive_nmf_k{k}.npz",
        W=W,
        H=H,
        labels=labels
    )


# =====================================
# 4. Save results
# =====================================

metrics_path = METRICS_DIR / "tanh_positive_nmf_results.json"

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\n===== Tanh NMF Completed =====")
print("Results saved to:", OUTPUT_DIR)
print("Metrics saved to:", metrics_path)
