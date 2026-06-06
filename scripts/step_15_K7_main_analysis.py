"""Step 15 - K=7 main analysis (raw fcluster, no C9 / noise merging).

Produces the headline K=7 macro typology profile used throughout
Sections 4 and 5 of paper 2. The C9 singleton (Iterative Repetition,
M3) is preserved as its own macro and the noise group (cluster_id=-1)
is profiled separately rather than absorbed.

Inputs:
  data/intermediate/actions_with_clusters.csv
  data/intermediate/macro_assignments.json   (K=7 -> raw_fcluster)

Outputs:
  data/final/macro_K7_raw.md       - full profile of the 7 macros
  data/final/noise_group_profile.md - profile of the noise group
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42
np.random.seed(SEED)

_HERE       = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(_HERE, "..", "..", "data", "Part7_outputs", "outputs")
ACTIONS_CSV = os.path.join(_HERE, "..", "..", "..", "shared", "action_decomposition", "actions_with_clusters.csv")
ASSIGN_JSON = os.path.join(OUT_DIR, "macro_assignments.json")

STANDARD_STAGES = ["Intent_Communication", "Navigation_Addressing",
                   "Perception_Diagnosis", "Manipulation_Execution",
                   "Feedback_Verification"]
STANDARD_TYPES  = ["Cognitive", "Physical", "mixed", "other"]


def normalize_stage(s):
    s = str(s)
    for std in STANDARD_STAGES:
        if std.lower() in s.lower():
            return std
    return STANDARD_STAGES[-1]


def normalize_type(t):
    t = str(t).lower().strip()
    if "cognitive" in t and "physical" in t:
        return "mixed"
    if "cognitive" in t:
        return "Cognitive"
    if "physical" in t:
        return "Physical"
    return "other"


def macro_discriminative_tfidf(group_texts_dict, top_n=15):
    """
    Compute TF-IDF where each macro is treated as ONE document.
    Returns {group_id: [(keyword, score), ...]} - keywords most distinctive
    of that macro vs the other macros.
    """
    group_ids = sorted(group_texts_dict.keys())
    docs = [" ".join(group_texts_dict[g]) for g in group_ids]
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_df=0.95,
        stop_words="english", lowercase=True, max_features=30000,
    )
    M = vec.fit_transform(docs)
    feats = np.array(vec.get_feature_names_out())
    out = {}
    for i, g in enumerate(group_ids):
        scores = np.asarray(M[i].todense()).flatten()
        idx = scores.argsort()[::-1][:top_n]
        out[g] = [(str(feats[j]), float(round(scores[j], 4))) for j in idx]
    return out


def internal_tfidf(texts, top_n=30):
    """TF-IDF inside one corpus, treating each row as a document.
    Top terms by mean TF-IDF across the corpus."""
    if not texts:
        return []
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.85,
        stop_words="english", lowercase=True, max_features=30000,
    )
    M = vec.fit_transform(texts)
    feats = np.array(vec.get_feature_names_out())
    mean_scores = np.asarray(M.mean(axis=0)).flatten()
    idx = mean_scores.argsort()[::-1][:top_n]
    return [(str(feats[j]), float(round(mean_scores[j], 4))) for j in idx]


def main():
    print(">>> Loading data ...")
    df = pd.read_csv(ACTIONS_CSV)
    with open(ASSIGN_JSON, encoding="utf-8") as f:
        assigns = json.load(f)

    raw_K7 = assigns["K7"]["raw_fcluster"]   # {"C0": 7, "C1": 6, ...}
    raw_K7 = {int(k[1:]): int(v) for k, v in raw_K7.items()}

    # Map each action to its macro (raw fcluster). Noise stays as -1.
    def map_macro(cid):
        if cid < 0:
            return -1
        return raw_K7.get(int(cid), -1)
    df["macro_id"] = df["cluster_id"].apply(map_macro)

    # ─────────────────────────────────────────────────────────────────────
    # Product 1: macro_K7_raw.md (7 macros, NO merging)
    # ─────────────────────────────────────────────────────────────────────
    print(">>> Building macro_K7_raw.md ...")
    macro_to_micros = {}
    for cid, mid in raw_K7.items():
        macro_to_micros.setdefault(mid, []).append(cid)

    macro_ids = sorted(macro_to_micros)

    df_clean = df[df["cluster_id"] >= 0].copy()
    total_clean = len(df_clean)

    # Compute discriminative TF-IDF (treating each macro as 1 document)
    group_texts = {
        mid: df_clean[df_clean["macro_id"] == mid]["action_description"]
                  .fillna("").astype(str).tolist()
        for mid in macro_ids
    }
    disc_tfidf = macro_discriminative_tfidf(group_texts, top_n=15)

    lines = []
    lines.append("# Macro-Cluster Analysis - K=7 (RAW fcluster, no singleton merging)")
    lines.append("")
    lines.append("Hierarchical clustering of 35 micro-clusters via Ward linkage on a 15-d feature vector "
                 "(5d UMAP centroid + 5d stage distribution + 4d cog/phys distribution + 1d avg step_order, all z-scored). "
                 "fcluster `t=7, criterion='maxclust'`.")
    lines.append("")
    lines.append(f"Total clustered actions (excluding noise): {total_clean:,} of 15,817 ({total_clean/15817*100:.1f}%)")
    lines.append("")
    lines.append("**Note**: One macro contains only C9 (the 'repeat steps' lone wolf). It is preserved as a separate macro because it represents a methodologically distinct meta-process pattern.")
    lines.append("")
    lines.append("**TF-IDF method**: Each macro is treated as one document; TF-IDF identifies keywords most distinctive of that macro relative to the other macros.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for mid in macro_ids:
        micros = sorted(macro_to_micros[mid])
        sub = df_clean[df_clean["macro_id"] == mid]
        n = len(sub)
        share = n / total_clean * 100
        share_global = n / 15817 * 100

        lines.append(f"## Macro-{mid}  ({len(micros)} micro-cluster{'s' if len(micros) > 1 else ''}, "
                     f"{n:,} actions, {share:.1f}% of clustered, {share_global:.1f}% of all 15,817)")
        lines.append("")
        lines.append(f"**Member micro-clusters:** {', '.join(f'C{c}' for c in micros)}")
        lines.append("")

        # Member micro sizes
        lines.append("**Member micro sizes:**")
        lines.append("")
        lines.append("| Micro | Size |")
        lines.append("|---|---|")
        for c in micros:
            sz = (df_clean["cluster_id"] == c).sum()
            lines.append(f"| C{c} | {sz} |")
        lines.append("")

        # TF-IDF Top-15 (discriminative)
        lines.append("**Top-15 TF-IDF keywords (discriminative, this macro vs the other 6):**")
        lines.append("")
        lines.append("| Keyword | Score |")
        lines.append("|---|---|")
        for k, s in disc_tfidf[mid]:
            lines.append(f"| `{k}` | {s:.4f} |")
        lines.append("")

        # Stage distribution
        stages_norm = sub["mapped_stage"].apply(normalize_stage)
        sd = stages_norm.value_counts(normalize=True)
        lines.append("**`mapped_stage` distribution:**")
        lines.append("")
        lines.append("| Stage | Share |")
        lines.append("|---|---|")
        for s in STANDARD_STAGES:
            lines.append(f"| `{s}` | {sd.get(s, 0)*100:.1f}% |")
        lines.append("")

        # Cog/phys distribution
        types_norm = sub["cognitive_or_physical"].apply(normalize_type)
        td = types_norm.value_counts(normalize=True)
        lines.append("**`cognitive_or_physical` distribution:**")
        lines.append("")
        lines.append("| Type | Share |")
        lines.append("|---|---|")
        for t in STANDARD_TYPES:
            lines.append(f"| {t} | {td.get(t, 0)*100:.1f}% |")
        lines.append("")

        # step_order
        so = sub["step_order"].astype(float)
        lines.append(f"**step_order**: mean = {so.mean():.2f}, median = {so.median():.1f}")
        lines.append("")

        # 15 random action_description
        sample_n = min(15, n)
        samples = sub.sample(n=sample_n, random_state=SEED)
        lines.append(f"**{sample_n} random `action_description` samples:**")
        lines.append("")
        for _, row in samples.iterrows():
            desc = str(row["action_description"]).replace("|", "\\|").strip()
            title = str(row["DWA_Title"])[:55].strip()
            lines.append(f"- _{title}_ → {desc}")
        lines.append("")

        # Top-5 most frequent DWA_IDs in this macro
        dwa_counts = sub.groupby(["DWA_ID", "DWA_Title"]).size().reset_index(name="n_steps_in_macro")
        dwa_counts = dwa_counts.sort_values("n_steps_in_macro", ascending=False).head(5)
        lines.append("**Top-5 DWAs by number of steps in this macro (most representative DWAs):**")
        lines.append("")
        lines.append("| DWA_ID | DWA_Title | # of steps in this macro |")
        lines.append("|---|---|---|")
        for _, row in dwa_counts.iterrows():
            t = str(row["DWA_Title"]).replace("|", "\\|").strip()
            lines.append(f"| `{row['DWA_ID']}` | {t} | {row['n_steps_in_macro']} |")
        lines.append("")

        lines.append("---")
        lines.append("")

    out_path = os.path.join(OUT_DIR, "macro_K7_raw.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   -> {out_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Product 2: noise_group_profile.md (cluster_id == -1)
    # ─────────────────────────────────────────────────────────────────────
    print(">>> Building noise_group_profile.md ...")
    noise = df[df["cluster_id"] == -1].copy()
    n_noise = len(noise)

    lines = []
    lines.append("# Noise Group Profile (cluster_id = -1)")
    lines.append("")
    lines.append("HDBSCAN labeled these actions as not belonging to any micro-cluster. They are **not** included in the K=7 macro analysis.")
    lines.append("")
    lines.append(f"**Total**: {n_noise:,} actions ({n_noise/15817*100:.1f}% of all 15,817)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # TF-IDF Top-30 (internal, treating each action as a document)
    lines.append("## TF-IDF Top-30 keywords (internal, each action as one document)")
    lines.append("")
    kws = internal_tfidf(noise["action_description"].fillna("").astype(str).tolist(), top_n=30)
    lines.append("| Rank | Keyword | Score |")
    lines.append("|---|---|---|")
    for i, (k, s) in enumerate(kws, 1):
        lines.append(f"| {i} | `{k}` | {s:.4f} |")
    lines.append("")

    # Stage distribution
    stages_norm = noise["mapped_stage"].apply(normalize_stage)
    sd = stages_norm.value_counts(normalize=True)
    lines.append("## `mapped_stage` distribution")
    lines.append("")
    lines.append("| Stage | Share |")
    lines.append("|---|---|")
    for s in STANDARD_STAGES:
        lines.append(f"| `{s}` | {sd.get(s, 0)*100:.1f}% |")
    lines.append("")

    # cog/phys
    types_norm = noise["cognitive_or_physical"].apply(normalize_type)
    td = types_norm.value_counts(normalize=True)
    lines.append("## `cognitive_or_physical` distribution")
    lines.append("")
    lines.append("| Type | Share |")
    lines.append("|---|---|")
    for t in STANDARD_TYPES:
        lines.append(f"| {t} | {td.get(t, 0)*100:.1f}% |")
    lines.append("")

    # step_order stats
    so = noise["step_order"].astype(float)
    lines.append("## `step_order` statistics")
    lines.append("")
    lines.append(f"- mean   = {so.mean():.2f}")
    lines.append(f"- median = {so.median():.1f}")
    lines.append(f"- min    = {int(so.min())}")
    lines.append(f"- max    = {int(so.max())}")
    lines.append("")

    # 30 random samples
    samples = noise.sample(n=min(30, n_noise), random_state=SEED)
    lines.append("## 30 random `action_description` samples")
    lines.append("")
    for _, row in samples.iterrows():
        desc = str(row["action_description"]).replace("|", "\\|").strip()
        title = str(row["DWA_Title"])[:55].strip()
        lines.append(f"- _{title}_ → {desc}")
    lines.append("")

    # Top-10 DWAs
    dwa_counts = noise.groupby(["DWA_ID", "DWA_Title"]).size().reset_index(name="n_steps_in_noise")
    dwa_counts = dwa_counts.sort_values("n_steps_in_noise", ascending=False).head(10)
    lines.append("## Top-10 DWAs by number of steps in the noise group")
    lines.append("")
    lines.append("| DWA_ID | DWA_Title | # noise steps |")
    lines.append("|---|---|---|")
    for _, row in dwa_counts.iterrows():
        t = str(row["DWA_Title"]).replace("|", "\\|").strip()
        lines.append(f"| `{row['DWA_ID']}` | {t} | {row['n_steps_in_noise']} |")
    lines.append("")

    out_path = os.path.join(OUT_DIR, "noise_group_profile.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   -> {out_path}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
