# -*- coding: utf-8 -*-
"""TEST 1: run the paper's pipeline on O*NET 30.2's native 18,796 task statements.
If the two-pole geometry and a substrate-like noise layer appear here too,
the micro-action decomposition is not necessary for the geometry."""
import pandas as pd, numpy as np, time
from scipy import stats
t0=time.time()
V=r"e:/大论文及4小论文/1_四篇小论文/论文2_Bipolar"
ts=pd.read_csv(f"{V}/decisive_tests/onet_30_2/Task Statements.txt",sep="\t",dtype=str)
ts.columns=[c.strip() for c in ts.columns]
tasks=ts.drop_duplicates("Task ID")[["Task ID","Task"]].reset_index(drop=True)
print(f"tasks: {len(tasks)}")
t2d=pd.read_csv(f"{V}/decisive_tests/onet_30_2/Tasks to DWAs.txt",sep="\t",dtype=str)
t2d.columns=[c.strip() for c in t2d.columns]
oai=pd.read_csv(f"{V}/dataset_zenodo_v1/06_dwa_automation_index_oai.csv")[["DWA_ID","Automation_Index"]].dropna()
# task-level OAI = mean over linked DWAs
tv=t2d.merge(oai,left_on="DWA ID",right_on="DWA_ID").groupby("Task ID")["Automation_Index"].mean()
tasks["oai"]=tasks["Task ID"].map(tv)
print(f"tasks with OAI: {tasks.oai.notna().sum()}")

from sentence_transformers import SentenceTransformer
m=SentenceTransformer("all-mpnet-base-v2")
emb=m.encode(tasks["Task"].tolist(),batch_size=64,show_progress_bar=True,normalize_embeddings=True)
np.save(f"{V}/decisive_tests/task_embeddings_mpnet.npy",emb)
print(f"embedded in {time.time()-t0:.0f}s")

import umap, hdbscan
red=umap.UMAP(n_components=5,n_neighbors=30,min_dist=0.0,metric="cosine",random_state=42).fit_transform(emb)
np.save(f"{V}/decisive_tests/task_umap5.npy",red)
cl=hdbscan.HDBSCAN(min_cluster_size=100,min_samples=20,cluster_selection_method="eom").fit(red)
lab=cl.labels_
tasks["hdb"]=lab
noise=(lab==-1).mean()
print(f"\nHDBSCAN: {lab.max()+1} clusters, noise/substrate share = {noise:.3f}  (paper: 0.356)")

# Ward K=7 over micro-cluster centroids (semantic-only, the clean variant)
from scipy.cluster.hierarchy import linkage, fcluster
cids=sorted(set(lab)-{-1})
cent=np.array([red[lab==c].mean(0) for c in cids])
Z=linkage((cent-cent.mean(0))/cent.std(0),method="ward")
for K in [7]:
    mac=fcluster(Z,K,criterion="maxclust")
    c2m=dict(zip(cids,mac))
    tasks["macro"]=tasks.hdb.map(c2m)
    sub=tasks.dropna(subset=["oai","macro"])
    g=sub.groupby("macro")["oai"]; mm=g.mean(); nn=g.size()
    keep=nn[nn>=20].index; mm=mm[keep]
    hi,lo=mm.idxmax(),mm.idxmin()
    a=sub[sub.macro==hi]["oai"].values; b=sub[sub.macro==lo]["oai"].values
    gt=sum((x>b).sum() for x in a); lt=sum((x<b).sum() for x in a); d=(gt-lt)/(len(a)*len(b))
    H,p=stats.kruskal(*[sub[sub.macro==k]["oai"].values for k in keep])
    # middle-pair non-significance (Bonferroni over all pairs among non-extreme classes)
    mids=[k for k in keep if k not in (hi,lo)]
    from itertools import combinations
    pairs=list(combinations(mids,2)); alpha=0.05/max(len(pairs),1); ns=0
    for x,y in pairs:
        _,pp=stats.mannwhitneyu(sub[sub.macro==x]["oai"],sub[sub.macro==y]["oai"])
        ns+= (pp>=alpha)
    print(f"K={K}: gap={mm.max()-mm.min():.3f}  extreme Cliff's d={d:+.3f}  H={H:.0f}  middle non-sig {ns}/{len(pairs)}")
    print("group means:",{int(k):round(v,3) for k,v in mm.sort_values().items()})
tasks.to_csv(f"{V}/decisive_tests/task_statement_clusters.csv",index=False)
print(f"\nTOTAL {time.time()-t0:.0f}s ; saved task_statement_clusters.csv")
