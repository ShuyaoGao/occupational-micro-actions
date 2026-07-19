# -*- coding: utf-8 -*-
"""Generic-substrate integrity audit against the annotator-documented filler mode.

Tests whether the HDBSCAN noise layer ("generic substrate", 35.6% of actions) is
enriched in the ceremonial/navigational filler steps that the three blind
annotators identified as a generation failure mode. Two lexicons are used:
(1) NARROW: built strictly from the annotators' own examples;
(2) BROAD:  deliberately broadened IN FAVOUR of the artefact hypothesis
    (the descriptor reports the BROAD-lexicon figures, the more conservative test).
A substrate depleted (OR < 1) even under the broad lexicon indicates the filler
mode does not manufacture the substrate.

Reported in the Data Descriptor (Technical Validation, "Generic-substrate
integrity"): substrate 5.2% vs clustered 6.7%, OR 0.76, Fisher P = 1.4e-4;
per-DWA Spearman rho = -0.099, P = 1.0e-5, n = 1,961.
"""
import pandas as pd, numpy as np
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

V = r"e:/大论文及4小论文/1_四篇小论文/论文2_Bipolar"
d = pd.read_csv(f"{V}/dataset_zenodo_v1/04_micro_actions_intelligence_types.csv",
                encoding="utf-8-sig")
d["action_description"] = d.action_description.astype(str)
S = (d.cluster_id == -1).values
nS, nC = int(S.sum()), int((~S).sum())

NARROW = {  # strictly from the annotators' free-text examples
 "greeting/farewell": r"\b(greet|say hello|says? goodbye|introduce (your|him|her|them)self|thank the (customer|client|patient)|bid farewell|exchange pleasantries)\b",
 "navigation padding": r"\b(walk to|drive to|travel to the (site|location)|open the (door|vehicle)|close the (door|vehicle)|get (in|out) of the|enter the (building|room|vehicle)|exit the (building|room|vehicle)|return to the (truck|vehicle|van))\b",
 "system login":       r"\b(log ?in(to)?|log ?on|sign in|launch the (system|application|software)|open the (application|program|software))\b",
 "tool retrieval":     r"\b(retrieve the|pick up the|gather the (tools|equipment|materials)|collect the (tools|equipment))\b",
}
BROAD = {  # deliberately generous to the "substrate is filler" hypothesis
 "greeting/farewell": r"(?:greet|say hello|goodbye|introduce yourself|introduce him|introduce her|introduce them|thank the|farewell|pleasantr)",
 "navigation/travel": r"(?:navigate to|walk to|drive to|travel to|proceed to|move to the|go to the|head to the|open the door|close the door|get in|get out of|enter the (?:building|room|vehicle|site)|exit the|return to the (?:truck|vehicle|van|office))",
 "system login/open": r"(?:log ?in|log ?on|sign in|launch the|open the (?:application|program|software|system))",
 "retrieval/fetch":   r"(?:retrieve the|pick up the|gather the|collect the (?:tools|equipment)|obtain the necessary)",
 "ceremonial close":  r"(?:confirm receipt|acknowledge the|express (?:thanks|appreciation)|close the (?:call|meeting|conversation))",
}

def audit(lex, name):
    print(f"\n===== {name} lexicon =====")
    print(f"{'pattern':20s} {'sub %':>7} {'clus %':>7} {'ratio':>7} {'Fisher p':>10}")
    for nm, pat in lex.items():
        h = d.action_description.str.contains(pat, case=False, regex=True, na=False).values
        a, b = int((h & S).sum()), int((h & ~S).sum())
        ps, pc = 100*a/nS, 100*b/nC
        _, pv = stats.fisher_exact([[a, nS-a], [b, nC-b]])
        print(f"{nm:20s} {ps:7.3f} {pc:7.3f} {ps/pc if pc else float('nan'):7.2f} {pv:10.2e}")
    anyf = d.action_description.str.contains("|".join(lex.values()),
                                             case=False, regex=True, na=False).values
    a, b = int((anyf & S).sum()), int((anyf & ~S).sum())
    ps, pc = 100*a/nS, 100*b/nC
    orr, pv = stats.fisher_exact([[a, nS-a], [b, nC-b]])
    print(f"ANY: substrate {a} ({ps:.2f}%) vs clustered {b} ({pc:.2f}%)  "
          f"ratio={ps/pc:.2f}  OR={orr:.2f}  p={pv:.2e}")
    return anyf

audit(NARROW, "NARROW (annotator-derived)")
anyf = audit(BROAD, "BROAD (artefact-hypothesis-favouring; reported in descriptor)")

# per-DWA: does substrate share track filler rate?
d["f"], d["s"] = anyf, S
g = d.groupby("DWA_ID").agg(sub_share=("s", "mean"), filler_rate=("f", "mean"), n=("s", "size"))
g = g[g.n >= 5]
r, p = stats.spearmanr(g.sub_share, g.filler_rate)
print(f"\nper-DWA (broad lexicon): substrate share vs filler rate  "
      f"rho={r:+.3f}  p={p:.2e}  (n={len(g)} DWAs)")
print("lexicon coverage: broad lexicon matches "
      f"{100*d.f.mean():.2f}% of all steps (paraphrased filler is invisible to both lexicons)")
