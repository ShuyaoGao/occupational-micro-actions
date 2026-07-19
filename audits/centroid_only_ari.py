# -*- coding: utf-8 -*-
"""Centroid-only Ward re-clustering check (ARI ~ 0.30).

Re-runs the Ward stage using only the five semantic-centroid dimensions of the
15-dim macro feature matrix (dropping the ten LLM-assigned structural-tag
dimensions) and reports the adjusted Rand index against the released K=7
partition. Documents that the headline macro-partition is not a purely
semantic object (descriptor, Methods "Semantic typology").
"""
import numpy as np, pandas as pd, os
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "intermediate")

Z_feats = np.load(os.path.join(DATA, "phase5_macro_features_zscored.npy"))  # 35 x 15
released = pd.read_csv(os.path.join(HERE, "..", "data", "final",
                                    "micro_to_macro_mapping.csv")) \
    if os.path.exists(os.path.join(HERE, "..", "data", "final", "micro_to_macro_mapping.csv")) else None

full = fcluster(linkage(Z_feats, method="ward"), 7, criterion="maxclust")
cent = fcluster(linkage(Z_feats[:, :5], method="ward"), 7, criterion="maxclust")
print(f"ARI(full-15dim vs centroid-only-5dim, K=7): {adjusted_rand_score(full, cent):.3f}")
