# -*- coding: utf-8 -*-
"""B档分析: B1弃用126条检查 / B3每职业基底占比 / B4 ω表导出 / B5职业层反转"""
import numpy as np, pandas as pd, os
from scipy import stats

W = r'e:\大论文及4小论文\5_原始数据流水线\完整工作区_含四篇数据_6月'
DZ = r'e:\大论文及4小论文\1_四篇小论文\论文2_Bipolar\dataset_zenodo_v1'
OUT = r'e:\大论文及4小论文\1_四篇小论文\论文2_Bipolar\manuscript_NHB\btier_outputs'
os.makedirs(OUT, exist_ok=True)

print('===== B1: 被弃126条 vs 保留1961条 =====')
oai = pd.read_csv(os.path.join(DZ, '06_dwa_automation_index_oai.csv'))
kept = set(pd.read_csv(os.path.join(DZ, '01_dwa_decomposition_sequences.csv')).DWA_ID)
oai['dropped'] = ~oai.DWA_ID.isin(kept)
d, k = oai[oai.dropped], oai[~oai.dropped]
print(f'dropped n={len(d)}, kept n={len(k)}')
print(f'mean OAI: dropped={d.Automation_Index.mean():.3f}, kept={k.Automation_Index.mean():.3f}')
ks = stats.ks_2samp(d.Automation_Index, k.Automation_Index)
mw = stats.mannwhitneyu(d.Automation_Index, k.Automation_Index)
print(f'KS D={ks.statistic:.3f} p={ks.pvalue:.3f} | MW p={mw.pvalue:.3f}')
print('dropped OAI dist:', d.Automation_Index.value_counts(normalize=True).round(3).to_dict())
print('kept OAI dist:', k.Automation_Index.value_counts(normalize=True).round(3).to_dict())

print('\n===== B3: 每职业基底占比分布 =====')
prof = pd.read_csv(os.path.join(W, r'paper4_github_ready\data\occupation_profiles.csv'))
sub = prof['main_Noise_share8'].dropna()
print(f'n occupations={len(sub)}; mean={sub.mean():.3f}, sd={sub.std():.3f}, CV={sub.std()/sub.mean():.3f}')
print(f'quartiles: {sub.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict()}')

print('\n===== B4: ω 表导出 (923职业 x 4智能类型) =====')
omega = prof[['onet_soc', 'title', 'main_llm_share', 'main_mp_share', 'main_emb_share', 'main_hb_share',
              'main_Noise_share8', 'n_unique_DWAs']].rename(columns={
    'main_llm_share': 'omega_Linguistic', 'main_mp_share': 'omega_MultimodalPerception',
    'main_emb_share': 'omega_Embodied', 'main_hb_share': 'omega_HumanBound',
    'main_Noise_share8': 'substrate_share'})
omega.to_csv(os.path.join(OUT, 'occupation_omega_table.csv'), index=False)
print(f'saved occupation_omega_table.csv ({len(omega)} rows)')
print('omega sums (should ~1):', (omega[['omega_Linguistic','omega_MultimodalPerception','omega_Embodied','omega_HumanBound']].sum(axis=1)).describe().round(3).loc[['mean','min','max']].to_dict())

print('\n===== B5: 职业层面反转检验 (FO原生单位) =====')
occ_oai = pd.read_csv(os.path.join(W, r'shared\oai\output_11_Occupation_Automation_Index.csv'))
fo = pd.read_csv(os.path.join(W, r'shared\external_indices\fo_oxford_martin\frey_osborne_original_702.csv'))
print('FO cols:', fo.columns.tolist()[:6], 'n=', len(fo))
# SOC 对齐: onet soc -> soc6 (去小数后缀)
occ_oai['soc6'] = occ_oai['O*NET-SOC Code'].str[:7]
foc = [c for c in fo.columns if 'soc' in c.lower() or 'SOC' in c][0]
fpc = [c for c in fo.columns if 'prob' in c.lower()][0]
fo['soc6'] = fo[foc].astype(str).str[:7]
oa = occ_oai.groupby('soc6')['OAI_Weighted'].mean().reset_index()
m = oa.merge(fo[['soc6', fpc]].groupby('soc6').mean().reset_index(), on='soc6', how='inner')
rho, p = stats.spearmanr(m['OAI_Weighted'], m[fpc])
print(f'occupation-level FO vs OAI: n={len(m)}, Spearman rho={rho:.3f}, p={p:.2e}')
# 也报告与 omega_Linguistic 的关系
mm = prof.copy(); mm['soc6'] = mm.onet_soc.str[:7]
mo = mm.groupby('soc6')['main_llm_share'].mean().reset_index().merge(fo[['soc6', fpc]].groupby('soc6').mean().reset_index(), on='soc6')
r2, p2 = stats.spearmanr(mo['main_llm_share'], mo[fpc])
print(f'occupation-level FO vs omega_Linguistic: n={len(mo)}, rho={r2:.3f}, p={p2:.2e}')
