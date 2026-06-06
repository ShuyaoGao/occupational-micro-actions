"""K=5 raw fcluster robustness cut (Appendix A).

Reports the raw Ward output at K=5 (the next informative cut below the
K=7 headline) without merging the C9 singleton. Output matches the
format of step_15 (K=7 main analysis) for direct comparison.

Output: data/robustness/macro_K5_raw.md
"""

import os
import json
import numpy as np
import pandas as pd
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


def main():
    df = pd.read_csv(ACTIONS_CSV)
    with open(ASSIGN_JSON, encoding="utf-8") as f:
        assigns = json.load(f)

    raw_K5 = assigns["K5"]["raw_fcluster"]
    raw_K5 = {int(k[1:]): int(v) for k, v in raw_K5.items()}

    def map_macro(cid):
        if cid < 0:
            return -1
        return raw_K5.get(int(cid), -1)
    df["macro_id"] = df["cluster_id"].apply(map_macro)

    macro_to_micros = {}
    for cid, mid in raw_K5.items():
        macro_to_micros.setdefault(mid, []).append(cid)
    macro_ids = sorted(macro_to_micros)

    df_clean = df[df["cluster_id"] >= 0].copy()
    total_clean = len(df_clean)

    group_texts = {
        mid: df_clean[df_clean["macro_id"] == mid]["action_description"]
                  .fillna("").astype(str).tolist()
        for mid in macro_ids
    }
    disc_tfidf = macro_discriminative_tfidf(group_texts, top_n=15)

    lines = []
    lines.append("# Macro-Cluster Analysis - K=5 (RAW fcluster, robustness backup)")
    lines.append("")
    lines.append("Robustness check for the K=7 main analysis. Same hierarchical clustering "
                 "(Ward linkage on 15-d feature vector) but cut at K=5 instead of K=7. "
                 "C9 again emerges as a singleton macro (the 'repeat steps' lone wolf).")
    lines.append("")
    lines.append(f"Total clustered actions (excluding noise): {total_clean:,} of 15,817 ({total_clean/15817*100:.1f}%)")
    lines.append("")
    lines.append("**This file is intended for the paper's appendix only.** Main analysis uses K=7.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for mid in macro_ids:
        micros = sorted(macro_to_micros[mid])
        sub = df_clean[df_clean["macro_id"] == mid]
        n = len(sub)
        share_clean = n / total_clean * 100
        share_global = n / 15817 * 100

        lines.append(f"## Macro-{mid}  ({len(micros)} micro-cluster{'s' if len(micros) > 1 else ''}, "
                     f"{n:,} actions, {share_clean:.1f}% of clustered, {share_global:.1f}% of all 15,817)")
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

        # TF-IDF Top-15
        lines.append("**Top-15 TF-IDF keywords (discriminative, this macro vs the other 4):**")
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

        # 15 random samples
        sample_n = min(15, n)
        samples = sub.sample(n=sample_n, random_state=SEED)
        lines.append(f"**{sample_n} random `action_description` samples:**")
        lines.append("")
        for _, row in samples.iterrows():
            desc = str(row["action_description"]).replace("|", "\\|").strip()
            title = str(row["DWA_Title"])[:55].strip()
            lines.append(f"- _{title}_ → {desc}")
        lines.append("")

        # Top-5 DWAs
        dwa_counts = sub.groupby(["DWA_ID", "DWA_Title"]).size().reset_index(name="n_steps_in_macro")
        dwa_counts = dwa_counts.sort_values("n_steps_in_macro", ascending=False).head(5)
        lines.append("**Top-5 DWAs by number of steps in this macro:**")
        lines.append("")
        lines.append("| DWA_ID | DWA_Title | # steps |")
        lines.append("|---|---|---|")
        for _, row in dwa_counts.iterrows():
            t = str(row["DWA_Title"]).replace("|", "\\|").strip()
            lines.append(f"| `{row['DWA_ID']}` | {t} | {row['n_steps_in_macro']} |")
        lines.append("")

        lines.append("---")
        lines.append("")

    out_path = os.path.join(OUT_DIR, "macro_K5_raw.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   -> {out_path}")


if __name__ == "__main__":
    main()
