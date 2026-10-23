# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

"""Post-analysis checks for the manuscript; does not modify primary outputs.

Run: .venv/Scripts/python.exe -X utf8 src/manuscript_checks.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from common import ROOT, CFG, eligible, named_wald, checked_variance
from association import fit, standardize, BASE
from first_project_statistics import slope_vector, linear

OUT = ROOT / 'outputs/manuscript_checks'
OUT.mkdir(parents=True, exist_ok=True)

def export(name, frame):
    frame.to_csv(OUT / (name + '.csv'), index=False)
    (OUT / (name + '.md')).write_text(frame.to_markdown(index=False), encoding='utf-8')

def joint(model, indices):
    test = named_wald(model, indices)
    return dict(chi2=float(test.statistic), df=len(indices), p=float(test.pvalue))

def run():
    d = pd.read_parquet(ROOT / 'data/processed/week_4.parquet')
    b = pd.read_parquet(ROOT / 'data/processed/base.parquet')
    keys=['id_student','code_module','code_presentation']
    raw=pd.read_csv(ROOT/'data/raw/studentInfo.csv').merge(
        pd.read_csv(ROOT/'data/raw/studentRegistration.csv'),on=keys,validate='one_to_one')
    compared=keys+['final_result','date_registration','date_unregistration']
    pd.testing.assert_frame_equal(raw[compared].sort_values(keys).reset_index(drop=True),
        b[compared].sort_values(keys).reset_index(drop=True),check_dtype=False)
    assert set(map(tuple,d[keys].to_numpy()))==set(map(tuple,b.loc[eligible(b,4),keys].to_numpy()))
    unknown = b.final_result.eq('Withdrawn') & b.date_unregistration.isna()
    u = d.final_result.eq('Withdrawn') & d.date_unregistration.isna()
    assert unknown.sum() == 93
    export('unknown_withdrawal_audit', b.loc[unknown].groupby(['code_module', 'imd_group']).agg(
        raw_n=('id_student', 'size'), students=('id_student', 'nunique'),
        registration_missing=('date_registration', lambda x: x.isna().sum())).reset_index())
    export('unknown_withdrawal_eligible', d.loc[u].groupby(['code_module', 'imd_group']).agg(
        n=('id_student', 'size'), mean_active_days=('active_days', 'mean')).reset_index())
    rows, gaps, slopes = [], [], []
    reuse = '--reuse-withdrawal' in sys.argv
    for label, frame in ([] if reuse else [('primary', d), ('exclude_unknown_withdrawal', d.loc[~u])]):
        for outcome in ['success', 'withdrawal']:
            m = fit(frame, outcome=outcome)
            ix = [t for t in m.params.index if ':' in t and 'imd_group' in t and 'Missing' not in t]
            rows.append(dict(analysis=label, outcome=outcome, n=len(frame), students=frame.id_student.nunique(),
                success_rate=frame.success.mean(), converged=m.converged, **joint(m, ix)))
            for g in ['0–30%', '30–70%', '70–100%']:
                beta,se,p = linear(m, slope_vector(m,g)*.25)
                slopes.append(dict(analysis=label, outcome=outcome, group=g, OR=np.exp(beta),
                    low=np.exp(beta-1.96*se), high=np.exp(beta+1.96*se)))
            if outcome == 'success':
                # Same primary reference population isolates the fitted-model change.
                for level in [.25,.5,.75]:
                    pa,ga=standardize(m,d,'70–100%',level)
                    pb,gb=standardize(m,d,'0–30%',level)
                    se=np.sqrt((ga-gb)@m.cov_params().to_numpy()@(ga-gb))
                    gaps.append(dict(analysis=label,active_days=level*28,gap=pa-pb,low=pa-pb-1.96*se,high=pa-pb+1.96*se))
    if not reuse:
        export('unknown_withdrawal_models',pd.DataFrame(rows))
        export('unknown_withdrawal_OR',pd.DataFrame(slopes))
        export('unknown_withdrawal_gaps',pd.DataFrame(gaps))
    print('Unknown withdrawal checks complete',flush=True)
    module_rows, module_slopes, loo = [], [], []
    reuse_modules = '--reuse-modules' in sys.argv
    for mod in ([] if reuse_modules else sorted(d.code_module.unique())):
        # Consistent known-IMD population across all seven module analyses.
        # GGG has only seven Missing-IMD records; the unrestricted Missing slope
        # produced a non-finite sandwich matrix in the initial diagnostic fit.
        a=d[d.code_module.eq(mod) & d.imd_group.ne('Missing')].copy()
        # Within-module model includes presentation fixed effects; remove unused module levels.
        for col in ['imd_group','context','gender','age_band','highest_education','disability','region']:
            a[col]=a[col].astype(str)
        m=fit(a)
        ix=[t for t in m.params.index if ':' in t and 'imd_group' in t and 'Missing' not in t]
        module_rows.append(dict(module=mod,n=len(a),students=a.id_student.nunique(),success_rate=a.success.mean(),
            converged=m.converged,**joint(m,ix)))
        for g in ['0–30%','30–70%','70–100%']:
            beta,se,p=linear(m,slope_vector(m,g)*.25)
            module_slopes.append(dict(module=mod,group=g,n=int(a.imd_group.eq(g).sum()),OR=np.exp(beta),low=np.exp(beta-1.96*se),high=np.exp(beta+1.96*se)))
        rem=d[~d.code_module.eq(mod)].copy()
        for col in ['imd_group','context','gender','age_band','highest_education','disability','region']:rem[col]=rem[col].astype(str)
        lm=fit(rem)
        lix=[t for t in lm.params.index if ':' in t and 'imd_group' in t and 'Missing' not in t]
        pa,ga=standardize(lm,rem,'70–100%',.75);pb,gb=standardize(lm,rem,'0–30%',.75)
        se=np.sqrt((ga-gb)@lm.cov_params().to_numpy()@(ga-gb))
        loo.append(dict(omitted_module=mod,n=len(rem),gap_21days=pa-pb,low=pa-pb-1.96*se,high=pa-pb+1.96*se,**joint(lm,lix)))
        print('Module',mod,'complete',flush=True)
    if not reuse_modules:
        mt=pd.DataFrame(module_rows);mt['posthoc_BH_q']=multipletests(mt.p,method='fdr_bh')[1]
        export('module_interactions',mt);export('module_OR',pd.DataFrame(module_slopes));export('leave_one_module_out_association',pd.DataFrame(loo))
    # Explicit reference-coded products avoid redundant module intercepts already
    # spanned by context. Formula expansion minus module mains still expands
    # IMD-by-module into full IMD indicators, causing six redundant columns.
    known=d[d.imd_group.ne('Missing')].copy()
    extra=[]
    for mod in sorted(known.code_module.unique())[1:]:
        flag=known.code_module.eq(mod).astype(float)
        name='module_slope_'+mod;known[name]=flag*known.active_proportion;extra.append(name)
        for j,g in enumerate(['30–70%','70–100%']):
            indicator=known.imd_group.eq(g).astype(float)
            name=f'module_imd_{mod}_{j}';known[name]=flag*indicator;extra.append(name)
            name=f'three_way_{mod}_{j}';known[name]=flag*indicator*known.active_proportion;extra.append(name)
    pooled=smf.glm('success ~ C(imd_group)*active_proportion + '+BASE+' + '+' + '.join(extra),
        data=known,family=sm.families.Binomial()).fit(maxiter=100,cov_type='cluster',cov_kwds={'groups':known.id_student.to_numpy()})
    assert np.linalg.matrix_rank(pooled.model.exog)==pooled.model.exog.shape[1]
    names=list(pooled.params.index)
    three=[t for t in names if t.startswith('three_way_')]
    # In the most-deprived reference group, are engagement slopes constant across modules?
    reference=[t for t in names if t.startswith('module_slope_')]
    pooled_tests=pd.DataFrame([
        dict(test='known_IMD_three_way',converged=pooled.converged,**joint(pooled,three)),
        dict(test='reference_group_engagement_by_module',converged=pooled.converged,**joint(pooled,reference))])
    pooled_tests['posthoc_BH_q']=multipletests(pooled_tests.p,method='fdr_bh')[1]
    export('pooled_module_tests',pooled_tests)
    additive=smf.glm('success ~ C(imd_group)+active_proportion + '+BASE,
        data=d,family=sm.families.Binomial()).fit(maxiter=100,cov_type='cluster',cov_kwds={'groups':d.id_student.to_numpy()})
    additive_gaps=[]
    for level in [.25,.5,.75]:
        pa,ga=standardize(additive,d,'70–100%',level);pb,gb=standardize(additive,d,'0–30%',level)
        se=np.sqrt((ga-gb)@additive.cov_params().to_numpy()@(ga-gb))
        additive_gaps.append(dict(active_days=level*28,gap=pa-pb,low=pa-pb-1.96*se,high=pa-pb+1.96*se))
    export('no_interaction_probability_gaps',pd.DataFrame(additive_gaps))
    tests=pd.read_csv(ROOT/'outputs/tables/T4_interaction_tests.csv')
    ix=tests.analysis.ne('primary') & tests.p.notna()
    q=multipletests(tests.loc[ix,'p'],method='fdr_bh')[1]
    assert np.allclose(q,tests.loc[ix,'exploratory_FDR_q'])
    splits=pd.read_csv(ROOT/'reports/split_assignments.csv')
    s=splits[(splits.week==4)&(splits.evaluation=='temporal')]
    sets={k:set(v.id_student) for k,v in s.groupby('split')}
    for x in sets:
        for y in sets:
            if x!=y:assert not sets[x]&sets[y]
    pred=pd.read_parquet(ROOT/'outputs/models/predictions.parquet')
    from sklearn.metrics import roc_auc_score
    perf=[]
    for block in ['B','C','D']:
        a=pred[(pred.week==4)&(pred.model=='Logistic')&(pred.block==block)&(pred.split=='2014_test')]
        perf.append(dict(block=block,n=len(a),students=a.id_student.nunique(),AUC=roc_auc_score(a.risk,a.probability)))
    export('verified_temporal_AUC',pd.DataFrame(perf))
    export('verified_split_sizes',s.groupby('split').agg(n=('id_student','size'),students=('id_student','nunique')).reset_index())
    manifest=dict(post_analysis=True,seed=CFG['seed'],raw_unknown_withdrawal=int(unknown.sum()),
        eligible_unknown_withdrawal=int(u.sum()),excluded_unknown_registration=int((unknown & ~eligible(b,4)).sum()),
        BH_family_size=int(ix.sum()),student_disjoint_splits=True,raw_registration_and_outcomes_verified=True,
        pooled_design_columns=pooled.model.exog.shape[1],pooled_design_rank=int(np.linalg.matrix_rank(pooled.model.exog)),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        week4_sha256=hashlib.sha256((ROOT/'data/processed/week_4.parquet').read_bytes()).hexdigest())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':run()
