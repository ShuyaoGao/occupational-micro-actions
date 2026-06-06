"""
Stage B - Semantic clustering of 15817 micro-actions.

Pipeline:
  1. Load Final_Golden_Dataset.csv -> flatten to action-level rows
  2. Embed action_description with sentence-transformers/all-mpnet-base-v2 (768d)
  3. UMAP -> 2d (visualize) and 5d (cluster)
  4. HDBSCAN on 5d
  5. Diagnostic profiles + plots

Outputs (under paper2_bipolar/data/Part7_outputs/outputs/):
  embeddings.npy
  umap_2d.npy, umap_5d.npy
  cluster_labels.npy
  actions_with_clusters.csv
  cluster_profiles.md
  umap_clusters.png
  cluster_similarity.png
  clustering_summary.json
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reproducibility
SEED = 42
np.random.seed(SEED)

_HERE   = os.path.dirname(os.path.abspath(__file__))
SRC_CSV = os.path.join(_HERE, "..", "..", "data", "Part5_outputs", "outputs", "Final_Golden_Dataset.csv")
OUT_DIR = os.path.join(_HERE, "..", "..", "data", "Part7_outputs", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Hyperparameters
EMBED_MODEL = "all-mpnet-base-v2"
EMBED_BATCH = 64

UMAP_2D_N_NEIGH, UMAP_2D_MIN_DIST = 30, 0.1
UMAP_5D_N_NEIGH, UMAP_5D_MIN_DIST = 30, 0.0

HDBSCAN_MIN_CLUSTER_SIZE = 100
HDBSCAN_MIN_SAMPLES      = 20


# -----------------------------------------------------------------------------
# 1. Load & flatten
# -----------------------------------------------------------------------------
def flatten_dataset():
    df_dwa = pd.read_csv(SRC_CSV)
    rows = []
    for _, r in df_dwa.iterrows():
        try:
            steps = json.loads(r["Final_Sequence_JSON"])
        except Exception:
            continue
        for s in steps:
            rows.append({
                "DWA_ID":              r["DWA_ID"],
                "DWA_Title":           r["DWA_Title"],
                "Resolution_Source":   r["Resolution_Source"],
                "step_order":          s.get("step_order"),
                "action_description":  s.get("action_description", ""),
                "mapped_stage":        s.get("mapped_stage", ""),
                "cognitive_or_physical": s.get("cognitive_or_physical", ""),
                "key_challenge":       s.get("key_challenge", ""),
            })
    flat = pd.DataFrame(rows)
    flat.insert(0, "action_id", range(1, len(flat) + 1))
    return flat


def normalize_type(t):
    t = str(t).lower().strip()
    if "cognitive" in t and "physical" in t:
        return "mixed"
    if "cognitive" in t:
        return "Cognitive"
    if "physical" in t:
        return "Physical"
    return "other"


# -----------------------------------------------------------------------------
# 2. Embed
# -----------------------------------------------------------------------------
def compute_embeddings(texts):
    cache = os.path.join(OUT_DIR, "embeddings.npy")
    if os.path.exists(cache):
        emb = np.load(cache)
        if emb.shape[0] == len(texts):
            print(f"  [cache] embeddings loaded: {emb.shape}")
            return emb
    print(f"  encoding {len(texts)} texts with {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(texts, show_progress_bar=True, batch_size=EMBED_BATCH,
                       convert_to_numpy=True, normalize_embeddings=False)
    np.save(cache, emb)
    print(f"  saved {cache}  shape={emb.shape}")
    return emb


# -----------------------------------------------------------------------------
# 3. UMAP
# -----------------------------------------------------------------------------
def run_umap(emb, n_components, n_neighbors, min_dist, cache_name):
    cache = os.path.join(OUT_DIR, cache_name)
    if os.path.exists(cache):
        out = np.load(cache)
        if out.shape == (emb.shape[0], n_components):
            print(f"  [cache] {cache_name} loaded: {out.shape}")
            return out
    print(f"  UMAP -> {n_components}d (n_neighbors={n_neighbors}, min_dist={min_dist}) ...")
    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
        metric="cosine", random_state=SEED,
    )
    out = reducer.fit_transform(emb)
    np.save(cache, out)
    print(f"  saved {cache}  shape={out.shape}")
    return out


# -----------------------------------------------------------------------------
# 4. HDBSCAN
# -----------------------------------------------------------------------------
def run_hdbscan(emb_5d):
    cache = os.path.join(OUT_DIR, "cluster_labels.npy")
    if os.path.exists(cache):
        labels = np.load(cache)
        if labels.shape[0] == emb_5d.shape[0]:
            print(f"  [cache] cluster_labels loaded: {labels.shape}")
            return labels
    print(f"  HDBSCAN (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, "
          f"min_samples={HDBSCAN_MIN_SAMPLES}) ...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method="eom",
        metric="euclidean",
    )
    labels = clusterer.fit_predict(emb_5d)
    np.save(cache, labels)
    return labels


# -----------------------------------------------------------------------------
# 5. Diagnostics
# -----------------------------------------------------------------------------
def cluster_summary(labels):
    """4.1 - overview stats."""
    counts = Counter(labels.tolist())
    n_total  = len(labels)
    n_noise  = counts.get(-1, 0)
    cluster_ids = sorted(c for c in counts if c >= 0)
    cluster_sizes = [counts[c] for c in cluster_ids]
    summary = {
        "n_total_actions": int(n_total),
        "n_clusters":      int(len(cluster_ids)),
        "n_noise":         int(n_noise),
        "noise_share":     round(n_noise / n_total, 4),
        "max_cluster_size":    int(max(cluster_sizes)) if cluster_sizes else 0,
        "min_cluster_size":    int(min(cluster_sizes)) if cluster_sizes else 0,
        "median_cluster_size": float(np.median(cluster_sizes)) if cluster_sizes else 0,
        "cluster_sizes":   {int(c): int(counts[c]) for c in cluster_ids},
        "hyperparameters": {
            "embed_model":        EMBED_MODEL,
            "umap_n_components_for_clustering": 5,
            "umap_n_neighbors":   UMAP_5D_N_NEIGH,
            "umap_min_dist":      UMAP_5D_MIN_DIST,
            "hdbscan_min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "hdbscan_min_samples":      HDBSCAN_MIN_SAMPLES,
            "random_state":       SEED,
        },
    }
    return summary


def tfidf_top_keywords(texts, all_texts, top_n=10):
    """Top-N TF-IDF keywords for a cluster, scored relative to the full corpus."""
    if len(texts) == 0:
        return []
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.85,
        stop_words="english", lowercase=True, max_features=20000,
    )
    vec.fit(all_texts)
    cluster_matrix = vec.transform(texts)
    mean_tfidf = np.asarray(cluster_matrix.mean(axis=0)).flatten()
    feats = np.array(vec.get_feature_names_out())
    top_idx = mean_tfidf.argsort()[::-1][:top_n]
    return [(str(feats[i]), float(round(mean_tfidf[i], 4))) for i in top_idx]


def cluster_profiles_md(df, labels, all_texts):
    """4.2 - per-cluster profile."""
    rng = np.random.RandomState(SEED)
    counts = Counter(labels.tolist())
    cluster_ids = [-1] + sorted(c for c in counts if c >= 0)

    lines = ["# Cluster Profiles", "",
             "Each cluster: size, 8 random samples, top-10 TF-IDF keywords, "
             "mapped_stage distribution, cognitive_or_physical distribution, "
             "and average action_description word count.", ""]

    df = df.copy()
    df["_cluster"]   = labels
    df["_norm_type"] = df["cognitive_or_physical"].apply(normalize_type)
    df["_word_count"] = df["action_description"].str.split().str.len()

    for cid in cluster_ids:
        sub = df[df["_cluster"] == cid]
        n   = len(sub)
        title = f"Noise (HDBSCAN -1)" if cid == -1 else f"Cluster {cid}"
        lines.append(f"## {title} - {n} samples ({n/len(df)*100:.1f}%)")
        lines.append("")

        # Random 8 samples
        sample_n = min(8, n)
        samples = sub.sample(n=sample_n, random_state=SEED) if n else sub
        lines.append("**Random samples:**")
        for _, row in samples.iterrows():
            desc = row["action_description"].replace("|", "\\|").strip()
            lines.append(f"- _{row['DWA_Title'][:60]}_  →  {desc}")
        lines.append("")

        # TF-IDF
        kw = tfidf_top_keywords(sub["action_description"].tolist(), all_texts, top_n=10)
        lines.append("**Top-10 TF-IDF keywords (cluster mean):**")
        lines.append("")
        lines.append("| Keyword | Score |")
        lines.append("|---|---|")
        for k, s in kw:
            lines.append(f"| `{k}` | {s:.4f} |")
        lines.append("")

        # Stage distribution
        sd = Counter(sub["mapped_stage"])
        lines.append("**`mapped_stage` distribution:**")
        lines.append("")
        lines.append("| Stage | Count | Share |")
        lines.append("|---|---|---|")
        for st, ct in sd.most_common(10):
            pct = ct / n * 100 if n else 0
            lines.append(f"| `{st}` | {ct} | {pct:.1f}% |")
        lines.append("")

        # Type distribution
        td = Counter(sub["_norm_type"])
        lines.append("**`cognitive_or_physical` (normalized) distribution:**")
        lines.append("")
        lines.append("| Type | Count | Share |")
        lines.append("|---|---|---|")
        for t, ct in td.most_common():
            pct = ct / n * 100 if n else 0
            lines.append(f"| {t} | {ct} | {pct:.1f}% |")
        lines.append("")

        # Word count
        avg_wc = sub["_word_count"].mean() if n else 0
        lines.append(f"**Average action_description word count:** {avg_wc:.1f}")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def plot_umap(emb_2d, labels, out_path):
    """4.3 - UMAP 2d scatter colored by cluster, noise in gray underneath."""
    counts = Counter(labels.tolist())
    cluster_ids = sorted(c for c in counts if c >= 0)
    n_clusters = len(cluster_ids)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=160)

    # Noise first (under)
    mask_noise = labels == -1
    if mask_noise.any():
        ax.scatter(emb_2d[mask_noise, 0], emb_2d[mask_noise, 1],
                   c="lightgray", s=4, alpha=0.45, linewidths=0,
                   label=f"noise ({mask_noise.sum()})")

    # Cluster points
    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "hsv")
    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        color = cmap(i % cmap.N) if n_clusters <= 20 else cmap(i / max(1, n_clusters))
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[color], s=8,
                   alpha=0.75, linewidths=0,
                   label=f"c{cid} ({mask.sum()})")

    ax.set_title(f"UMAP 2D - HDBSCAN clusters ({n_clusters} clusters, "
                 f"{mask_noise.sum()} noise)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, framealpha=0.9, ncol=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_similarity(emb_5d, labels, out_path):
    """4.4 - cosine similarity heatmap of cluster centroids (5d)."""
    counts = Counter(labels.tolist())
    cluster_ids = sorted(c for c in counts if c >= 0)
    centroids = np.vstack([
        emb_5d[labels == cid].mean(axis=0) for cid in cluster_ids
    ])
    sim = cosine_similarity(centroids)

    fig, ax = plt.subplots(figsize=(max(6, len(cluster_ids) * 0.5),
                                     max(5, len(cluster_ids) * 0.45)),
                           dpi=160)
    sns.heatmap(sim, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=[f"c{c}" for c in cluster_ids],
                yticklabels=[f"c{c}" for c in cluster_ids],
                ax=ax, cbar_kws={"label": "cosine similarity"},
                vmin=-1, vmax=1, square=True, linewidths=0.5)
    ax.set_title("Cluster centroid similarity (5d UMAP space)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print(">>> Step 1: load + flatten ...")
    df = flatten_dataset()
    print(f"    rows = {len(df):,}")

    texts = df["action_description"].fillna("").astype(str).tolist()

    print(">>> Step 2: embeddings ...")
    embeddings = compute_embeddings(texts)

    print(">>> Step 3a: UMAP 2d ...")
    umap_2d = run_umap(embeddings, 2, UMAP_2D_N_NEIGH, UMAP_2D_MIN_DIST, "umap_2d.npy")

    print(">>> Step 3b: UMAP 5d ...")
    umap_5d = run_umap(embeddings, 5, UMAP_5D_N_NEIGH, UMAP_5D_MIN_DIST, "umap_5d.npy")

    print(">>> Step 4: HDBSCAN ...")
    labels = run_hdbscan(umap_5d)
    counts = Counter(labels.tolist())
    n_noise = counts.get(-1, 0)
    n_clusters = len([c for c in counts if c >= 0])
    print(f"    {n_clusters} clusters, {n_noise} noise ({n_noise/len(labels)*100:.1f}%)")

    # Save labeled CSV
    df["cluster_id"] = labels
    out_csv = os.path.join(OUT_DIR, "actions_with_clusters.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"    -> {out_csv}")

    print(">>> Step 5: diagnostics ...")
    summary = cluster_summary(labels)
    with open(os.path.join(OUT_DIR, "clustering_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"    -> clustering_summary.json")

    md = cluster_profiles_md(df, labels, texts)
    with open(os.path.join(OUT_DIR, "cluster_profiles.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"    -> cluster_profiles.md")

    plot_umap(umap_2d, labels, os.path.join(OUT_DIR, "umap_clusters.png"))
    print(f"    -> umap_clusters.png")

    plot_cluster_similarity(umap_5d, labels, os.path.join(OUT_DIR, "cluster_similarity.png"))
    print(f"    -> cluster_similarity.png")

    print("\n[Done]")


if __name__ == "__main__":
    main()
