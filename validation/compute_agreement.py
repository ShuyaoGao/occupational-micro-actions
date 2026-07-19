# -*- coding: utf-8 -*-
"""标注回收后计算一致率 (自动识别 2 或 3 位标注者):
   A 拆解质量: 多标注者 ICC + 均分
   B 智能类型: 人-人 Fleiss/Cohen κ (关键缺口) + 各人 vs 模型 κ
用法: 把填好的 xlsx 放回本目录后运行 python compute_agreement.py"""
import numpy as np, pandas as pd, os, glob

D = os.path.dirname(os.path.abspath(__file__))

def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    labels = sorted(set(a) | set(b)); idx = {l: i for i, l in enumerate(labels)}
    n = len(a); m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b): m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = sum((m[i].sum()/n)*(m[:, i].sum()/n) for i in range(len(labels)))
    return (po - pe)/(1 - pe) if pe < 1 else 1.0

def fleiss_kappa(mat):  # mat: n_items x n_categories 计数
    N, k = mat.shape; nR = mat[0].sum()
    p = mat.sum(0)/(N*nR)
    P = ((mat**2).sum(1)-nR)/(nR*(nR-1))
    Pbar = P.mean(); Pe = (p**2).sum()
    return (Pbar-Pe)/(1-Pe) if Pe < 1 else 1.0

# ================= B: 智能类型 =================
bf = sorted(glob.glob(os.path.join(D, 'sampleB_*标注者*_已标注.xlsx')))
filled = []
norm = {'L': 'LLM', 'MP': 'Multimodal_Perception', 'E': 'Embodied', 'HB': 'Human_Bound'}
for f in bf:
    df = pd.read_excel(f); lc = [c for c in df.columns if 'label' in c.lower()][0]
    lab = df[lc].astype(str).str.strip().str.upper().map(lambda x: norm.get(x, x))
    if lab.notna().sum() >= 100 and (lab != 'NAN').sum() >= 100:
        filled.append(lab.values)
print('=== B 智能类型 ===')
if len(filled) >= 2:
    key = pd.read_csv(os.path.join(D, 'sampleB_KEY_model_labels.csv')).intelligence_type_A.values
    cats = ['LLM', 'Multimodal_Perception', 'Embodied', 'Human_Bound']
    ci = {c: i for i, c in enumerate(cats)}
    if len(filled) >= 3:
        cnt = np.zeros((len(filled[0]), 4))
        for lab in filled:
            for r, x in enumerate(lab):
                if x in ci: cnt[r, ci[x]] += 1
        print(f'人-人 Fleiss κ ({len(filled)}位, 关键): {fleiss_kappa(cnt):.3f}')
    ks = [cohen_kappa(filled[i], filled[j]) for i in range(len(filled)) for j in range(i+1, len(filled))]
    print(f'人-人 两两 Cohen κ 均值: {np.mean(ks):.3f}  (各: {[round(k,3) for k in ks]})')
    for i, lab in enumerate(filled, 1):
        print(f'  标注者{i} vs 模型 κ: {cohen_kappa(lab, key):.3f}')
    print('  >> 人-人 κ>0.6 即证明四分类是客观可复现构念; 用此数替换原 encoder-vs-authors κ=0.893')
else:
    print(f'待回收 (已找到 {len(filled)} 份填好, 需≥2)')

# ================= A: 拆解质量 =================
def icc(cols):
    M = np.column_stack(cols).astype(float); n, k = M.shape; gm = M.mean()
    msr = k*((M.mean(1)-gm)**2).sum()/(n-1)
    msc = n*((M.mean(0)-gm)**2).sum()/(k-1)
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0) + gm)**2).sum()/((n-1)*(k-1))
    return (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)

af = sorted(glob.glob(os.path.join(D, 'sampleA_*标注者*_已标注.xlsx')))
dfs = []
for f in af:
    df = pd.read_excel(f)
    qs = [c for c in df.columns if c.startswith('Q')]
    if any(pd.to_numeric(df[q], errors='coerce').notna().sum() >= 50 for q in qs):
        dfs.append(df)
print('\n=== A 拆解质量 ===')
if len(dfs) >= 2:
    qs = [c for c in dfs[0].columns if c.startswith('Q')]
    for q in qs:
        cols = [pd.to_numeric(d[q], errors='coerce') for d in dfs]
        m = np.all([c.notna().values for c in cols], axis=0)
        cc = [c[m].values for c in cols]
        print(f'{q}: 均分 {[round(float(c.mean()),2) for c in cc]}, ICC={icc(cc):.3f}, n={m.sum()}')
    print('  >> 三项均分>=4 且 ICC>=0.5 即可写"独立盲标者判定拆解完整/忠实/粒度合理, 人-人一致"')
else:
    print(f'待回收 (已找到 {len(dfs)} 份填好, 需≥2)')
