# -*- coding: utf-8 -*-
"""Deduplicated variant of the task-layer comparison: cluster the 17,537 unique
task TEXTS (referee point: 1,259 duplicate strings inflate local density and
could deflate the noise share). Same pipeline settings as test1."""
import pandas as pd, numpy as np, time
t0 = time.time()
V = r"e:/大论文及4小论文/1_四篇小论文/论文2_Bipolar"
ts = pd.read_csv(f"{V}/decisive_tests/onet_30_2/Task Statements.txt", sep="\t", dtype=str)
ts.columns = [c.strip() for c in ts.columns]
texts = sorted(set(ts["Task"].astype(str)))
print(f"unique task texts: {len(texts)}")

from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-mpnet-base-v2")
emb = m.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
print(f"embedded in {time.time()-t0:.0f}s")

import umap, hdbscan
red = umap.UMAP(n_components=5, n_neighbors=30, min_dist=0.0, metric="cosine",
                random_state=42).fit_transform(emb)
cl = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20,
                     cluster_selection_method="eom").fit(red)
lab = cl.labels_
print(f"DEDUP task layer: {lab.max()+1} clusters, noise share = {(lab==-1).mean():.3f}")
print(f"(with-duplicates run: 3 clusters, 3.3%; micro-action layer: 35 clusters, 35.6%)")
print(f"TOTAL {time.time()-t0:.0f}s")
