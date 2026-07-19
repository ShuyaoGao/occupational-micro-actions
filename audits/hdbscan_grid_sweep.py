# -*- coding: utf-8 -*-
"""HDBSCAN hyperparameter grid for the generic-substrate share.

Sweeps min_cluster_size x min_samples over the cached 5-d UMAP embedding and
records the noise (substrate) share. Result (hdbscan_grid.csv): the share stays
within 0.33-0.41 across the grid, except one degenerate cell
(min_cluster_size=200, min_samples=30) that collapses to two clusters.
Note: the grid varies the clusterer over a FIXED embedding realisation
(seed 42); it does not bound seed sensitivity.
"""
import numpy as np, pandas as pd, hdbscan, os

HERE = os.path.dirname(os.path.abspath(__file__))
UMAP5 = os.path.join(HERE, "..", "data", "intermediate", "umap_5d.npy")  # adjust if needed

X = np.load(UMAP5)
rows = []
for mcs in [50, 75, 100, 150, 200]:
    for ms in [10, 20, 30]:
        cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                             cluster_selection_method="eom").fit_predict(X)
        rows.append({"min_cluster_size": mcs, "min_samples": ms,
                     "n_clusters": int(cl.max()) + 1,
                     "noise_share": round(float((cl == -1).mean()), 3)})
        print(rows[-1])
pd.DataFrame(rows).to_csv(os.path.join(HERE, "hdbscan_grid.csv"), index=False)
print("saved hdbscan_grid.csv")
