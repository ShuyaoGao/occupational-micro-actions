# -*- coding: utf-8 -*-
"""Deep, honest analysis of the 3-rater blind annotation.
Writes a UTF-8 report to analysis_report.txt (console mojibake-safe)."""
import numpy as np, pandas as pd, os, glob, io
from collections import Counter

D = r"e:\大论文及4小论文\1_四篇小论文\论文2_Bipolar\人工验证"
out = io.StringIO()
def P(*a): print(*a, file=out)

norm = {'L':'LLM','MP':'Multimodal_Perception','E':'Embodied','HB':'Human_Bound'}
cats = ['LLM','Multimodal_Perception','Embodied','Human_Bound']
short = {'LLM':'L','Multimodal_Perception':'MP','Embodied':'E','Human_Bound':'HB'}

def cohen(a,b):
    a,b=np.asarray(a),np.asarray(b)
    labs=sorted(set(a)|set(b)); idx={l:i for i,l in enumerate(labs)}
    n=len(a); m=np.zeros((len(labs),len(labs)))
    for x,y in zip(a,b): m[idx[x],idx[y]]+=1
    po=np.trace(m)/n
    pe=sum((m[i].sum()/n)*(m[:,i].sum()/n) for i in range(len(labs)))
    return (po-pe)/(1-pe) if pe<1 else 1.0

# ===== Sample B =====
bf = sorted(glob.glob(os.path.join(D,'sampleB_*标注者*_已标注.xlsx')))
labels=[]
for f in bf:
    df=pd.read_excel(f)
    lc=[c for c in df.columns if 'label' in c.lower()][0]
    lab=df[lc].astype(str).str.strip().str.upper().map(lambda x:norm.get(x,x))
    labels.append(lab.values)
A1,A2,A3 = labels
key=pd.read_csv(os.path.join(D,'sampleB_KEY_model_labels.csv'))
model=key.intelligence_type_A.values
conf=key.confidence_A.values
desc=key.action_description.values
n=len(A1)

P("="*70)
P("SAMPLE B — INTELLIGENCE TYPE (4-class, n=%d)"%n)
P("="*70)

# human majority / unanimity
human_maj=[]; unanimous=[]
for i in range(n):
    votes=[A1[i],A2[i],A3[i]]
    c=Counter(votes); top,tc=c.most_common(1)[0]
    human_maj.append(top if tc>=2 else None)   # None = 3-way split (impossible w/4 cats & 3 raters unless all diff)
    unanimous.append(tc==3)
n_unan=sum(unanimous)
P("Unanimous (all 3 identical): %d / %d = %.1f%%"%(n_unan,n,100*n_unan/n))
n_maj=sum(1 for m in human_maj if m is not None)
P("At least 2/3 agree: %d / %d = %.1f%%"%(n_maj,n,100*n_maj/n))
n_split=n-n_maj
P("3-way split (no majority): %d"%n_split)

# per-category distribution
P("\nLabel distribution:")
P("  %-24s %5s %5s %5s %5s"%("category","A1","A2","A3","model"))
for c in cats:
    P("  %-24s %5d %5d %5d %5d"%(c,(A1==c).sum(),(A2==c).sum(),(A3==c).sum(),(model==c).sum()))

# model vs human-consensus
# consider only items with a human majority
P("\n--- MODEL vs HUMAN CONSENSUS ---")
maj_idx=[i for i in range(n) if human_maj[i] is not None]
agree_maj=sum(1 for i in maj_idx if model[i]==human_maj[i])
P("Model matches human-majority label: %d / %d = %.1f%%"%(agree_maj,len(maj_idx),100*agree_maj/len(maj_idx)))
unan_idx=[i for i in range(n) if unanimous[i]]
agree_unan=sum(1 for i in unan_idx if model[i]==A1[i])
P("Model matches UNANIMOUS human label: %d / %d = %.1f%%"%(agree_unan,len(unan_idx),100*agree_unan/len(unan_idx)))
P("  >> The %d unanimous-but-model-disagrees items are model-error candidates (humans converged, model differs)."%(len(unan_idx)-agree_unan))

# confusion where all 3 humans agree but model differs
P("\n--- MODEL-ERROR CANDIDATES (all 3 humans agree, model disagrees) ---")
P("These are the strongest 'humans better than AI' cases + v2 fix targets.\n")
errcases=[]
for i in unan_idx:
    if model[i]!=A1[i]:
        errcases.append((i,A1[i],model[i],conf[i],desc[i]))
# sort by model confidence descending (high-confidence model errors are most telling)
errcases.sort(key=lambda x:-x[3])
P("Total: %d items\n"%len(errcases))
for i,h,m,cf,d in errcases:
    P("  #%d  human=%s  model=%s  model_conf=%.2f"%(i+1,short[h],short[m],cf))
    P("      %s"%d)

# directional confusion: model_label -> human_unanimous_label
P("\n--- Directional confusion on model-error candidates (model->human) ---")
dirc=Counter((short[m],short[h]) for i,h,m,cf,d in errcases)
for (m,h),ct in dirc.most_common():
    P("  model said %-3s -> humans said %-3s : %d"%(m,h,ct))

# Also: confusion matrix human-majority (rows) vs model (cols)
P("\n--- Full confusion: human-majority (row) x model (col), majority items only ---")
cm=pd.DataFrame(0,index=[short[c] for c in cats],columns=[short[c] for c in cats])
for i in maj_idx:
    cm.loc[short[human_maj[i]],short[model[i]]]+=1
P(cm.to_string())

# per annotator vs model
P("\n--- Per-annotator vs model Cohen kappa ---")
for i,(nm,lab) in enumerate(zip(['A1','A2','A3'],labels),1):
    P("  %s vs model: kappa=%.3f, raw agree=%.1f%%"%(nm,cohen(lab,model),100*(lab==model).mean()))
P("  human-human Fleiss kappa = 0.899 (computed separately)")

# ===== Sample A =====
P("\n"+"="*70)
P("SAMPLE A — DECOMPOSITION QUALITY (79 DWAs, 3 items 1-5)")
P("="*70)
af=sorted(glob.glob(os.path.join(D,'sampleA_*标注者*_已标注.xlsx')))
dfsA=[pd.read_excel(f) for f in af]
qs=[c for c in dfsA[0].columns if c.startswith('Q')]
def icc(cols):
    M=np.column_stack(cols).astype(float); nn,k=M.shape; gm=M.mean()
    msr=k*((M.mean(1)-gm)**2).sum()/(nn-1)
    msc=nn*((M.mean(0)-gm)**2).sum()/(k-1)
    mse=((M-M.mean(1,keepdims=True)-M.mean(0)+gm)**2).sum()/((nn-1)*(k-1))
    return (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/nn)
for q in qs:
    cols=[pd.to_numeric(d[q],errors='coerce') for d in dfsA]
    m=np.all([c.notna().values for c in cols],axis=0)
    cc=[c[m].values for c in cols]
    M=np.column_stack(cc)
    # complementary ceiling-aware stats
    exact=np.mean([len(set(row))==1 for row in M])   # all 3 identical
    within1=np.mean([row.max()-row.min()<=1 for row in M])
    pct_ge4_mean=(M.mean(1)>=4).mean()
    pct_any_le2=(M.min(1)<=2).mean()
    P("\n%s"%q)
    P("  means per annotator: %s"%[round(float(c.mean()),2) for c in cc])
    P("  overall mean=%.2f  SD=%.2f  min=%d"%(M.mean(),M.std(),int(M.min())))
    P("  ICC(2,1)=%.3f  [low ICC here = ceiling effect, see exact-agree]"%icc(cc))
    P("  exact 3-way agreement: %.1f%%   within-1: %.1f%%"%(100*exact,100*within1))
    P("  items with mean>=4: %.1f%%   items any rater<=2: %.1f%%"%(100*pct_ge4_mean,100*pct_any_le2))

# lowest-scoring items (where annotators flagged problems)
P("\n--- LOWEST-SCORING DECOMPOSITIONS (mean across 3 annotators, ascending) ---")
P("(these are where annotators saw the most problems)\n")
base=dfsA[0][['item','DWA_ID','DWA_title']].copy()
qmat={}
for q in qs:
    qmat[q]=np.column_stack([pd.to_numeric(d[q],errors='coerce').values for d in dfsA]).mean(1)
base['Q1']=qmat[qs[0]]; base['Q2']=qmat[qs[1]]; base['Q3']=qmat[qs[2]]
base['mean3']=base[['Q1','Q2','Q3']].mean(1)
low=base.sort_values('mean3').head(12)
for _,r in low.iterrows():
    P("  #%d [%s] Q1=%.1f Q2=%.1f Q3=%.1f (mean=%.2f)"%(r['item'],r['DWA_ID'],r['Q1'],r['Q2'],r['Q3'],r['mean3']))
    P("      %s"%r['DWA_title'])

# ===== ALL NOTES =====
P("\n"+"="*70)
P("ALL ANNOTATOR NOTES (Sample A)")
P("="*70)
for ai,d in enumerate(dfsA,1):
    ncol=[c for c in d.columns if c.lower().startswith('note')]
    if not ncol: continue
    ncol=ncol[0]
    P("\n----- Annotator %d notes -----"%ai)
    cnt=0
    for _,r in d.iterrows():
        note=r[ncol]
        if pd.notna(note) and str(note).strip() and str(note).strip().lower()!='nan':
            cnt+=1
            P("  #%s [%s] %s"%(r['item'],r['DWA_title'],str(note).strip()))
    P("  (total %d notes)"%cnt)

# Sample B notes if any
P("\n----- Sample B notes (if present) -----")
for ai,f in enumerate(bf,1):
    d=pd.read_excel(f)
    ncol=[c for c in d.columns if c.lower().startswith('note')]
    if not ncol:
        P("  Annotator %d: no notes column"%ai); continue
    ncol=ncol[0]; cnt=0
    for _,r in d.iterrows():
        note=r[ncol]
        if pd.notna(note) and str(note).strip() and str(note).strip().lower()!='nan':
            cnt+=1
            P("  A%d #%s %s"%(ai,r.get('item','?'),str(note).strip()))
    if cnt==0: P("  Annotator %d: notes column empty"%ai)

with open(os.path.join(r"C:\Users\shuya\AppData\Local\Temp\claude\e------4---\6e48fa8a-3076-4f86-8e6a-679a1b029cb2\scratchpad","analysis_report.txt"),"w",encoding="utf-8") as fh:
    fh.write(out.getvalue())
print("written")
