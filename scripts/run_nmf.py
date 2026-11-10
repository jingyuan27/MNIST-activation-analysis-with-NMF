
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


# =====================================
# 1. Configuration
# =====================================

ROOT = Path(__file__).resolve().parents[1]

ACTIVATION_DIR = ROOT / "reports" / "activations"
OUTPUT_DIR = ROOT / "reports" / "nmf"
FIG_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. Load ReLU activations
# =====================================

activation_path = ACTIVATION_DIR / "relu_activations.npz"

if not activation_path.exists():
    raise FileNotFoundError(
        f"Activation file not found: {activation_path}"
    )

with np.load(activation_path) as data:
    A = data["activations"].astype(np.float64)
    labels = data["labels"]

print("===== Activation Matrix =====")
print("Shape:", A.shape)
print("Minimum:", A.min())
print("Maximum:", A.max())

# NMF requires nonnegative input
assert np.all(A >= 0), "NMF requires nonnegative data!"

assert A.ndim == 2, "Activation matrix must be 2-dimensional."


# =====================================
# 3. NMF decomposition
# =====================================

ranks = [4, 8, 12, 16, 24, 32, 48, 64]

results = []

for k in ranks:

    print(f"\n===== Running NMF: k={k} =====")

    model = NMF(
        n_components=k,
        init="nndsvda",
        solver="cd",
        max_iter=500,
        random_state=42,
    )

    start_time = time.time()

    W = model.fit_transform(A)
    H = model.components_

    elapsed = time.time() - start_time

    # Relative reconstruction error
    relative_error = (
        np.linalg.norm(A - W @ H, ord="fro")
        / np.linalg.norm(A, ord="fro")
    )

    result = {
        "rank": k,
        "reconstruction_error": float(relative_error),
        "iterations": int(model.n_iter_),
        "time_seconds": float(elapsed),
    }

    results.append(result)

    print("W shape:", W.shape)
    print("H shape:", H.shape)
    print("Relative error:", relative_error)
    print("Iterations:", model.n_iter_)
    print("Time:", round(elapsed, 2), "seconds")

    # Save factor matrices
    np.savez_compressed(
        OUTPUT_DIR / f"relu_nmf_k{k}.npz",
        W=W,
        H=H,
    )


# =====================================
# 4. Save numerical results
# =====================================

metrics_path = METRICS_DIR / "nmf_rank_comparison.json"

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)


# =====================================
# 5. Plot reconstruction error
# =====================================

plt.figure(figsize=(8, 5))

plt.plot(
    ranks,
    [r["reconstruction_error"] for r in results],
    marker="o",
)

plt.xlabel("NMF Rank (k)")
plt.ylabel("Relative Reconstruction Error")
plt.title("NMF Reconstruction Error vs Rank")
plt.xticks(ranks)
plt.grid(alpha=0.3)

plt.savefig(
    FIG_DIR / "nmf_rank_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("\n===== NMF Experiments Completed =====")
print("Results saved to:", OUTPUT_DIR)
print("Metrics saved to:", metrics_path)
