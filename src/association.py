# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

import numpy as np,pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy.special import expit
from statsmodels.stats.multitest import multipletests
from common import *
BASE='C(gender)+C(age_band)+C(highest_education)+C(disability)+C(region)+num_of_prev_attempts+studied_credits+C(context)'
def fit(df,exposure='active_proportion',outcome='success',quadratic=False,extra=''):
    term=f'({exposure}+I({exposure}**2))' if quadratic else exposure
    formula=f'{outcome} ~ C(imd_group)*{term} + '+BASE+extra
    model=smf.glm(formula,data=df,family=sm.families.Binomial()).fit(maxiter=100,cov_type='cluster',cov_kwds={'groups':df.id_student.to_numpy()})
    return validate_inference(model)
def standardize(model,df,group,level):
    d=df.copy();d['imd_group']=group;d['active_proportion']=level
    spec=model.model.data.model_spec
    x=np.asarray(spec.get_model_matrix(d) if hasattr(spec,'get_model_matrix') else build_design_matrices([spec],d)[0]);p=expit(x@model.params)
    grad=(x*(p*(1-p))[:,None]).mean(axis=0)
    return p.mean(),grad
def run():
    df=pd.read_parquet(ROOT/'data/processed/week_4.parquet')
    cats=['gender','age_band','highest_education','disability','region','imd_group','context']
    df[cats]=df[cats].fillna('Missing').astype(str)
    characteristics=[]
    for col in ['gender','age_band','highest_education','disability','region','imd_band']:
        for (g,v),a in df.groupby(['imd_group',col],dropna=False):characteristics.append(dict(group=g,variable=col,level=str(v),n=len(a),denominator=int(df.imd_group.eq(g).sum()),proportion=len(a)/df.imd_group.eq(g).sum()))
    table('T2_baseline',pd.DataFrame(characteristics))
    for col in ['num_of_prev_attempts','studied_credits']:table('T2_'+col,df.groupby('imd_group')[col].agg(['count','mean','std','median']).reset_index())
    summaries=df.groupby('imd_group').agg(enrolments=('success','size'),unique_students=('id_student','nunique'),successes=('success','sum'),success_rate=('success','mean'),mean_active_proportion=('active_proportion','mean'),median_clicks=('total_clicks','median'),zero_activity=('total_clicks',lambda s:s.eq(0).sum())).reset_index()
    masks={g:df.imd_group.eq(g).to_numpy() for g in summaries.imd_group}
    boot=student_descriptive_bootstrap(df,masks)
    summaries['ci_low']=[boot[g+'_success'].quantile(.025) for g in summaries.imd_group]
    summaries['ci_high']=[boot[g+'_success'].quantile(.975) for g in summaries.imd_group]
    summaries['ci_method']='student-cluster percentile bootstrap (1000; seed=20261005)'
    table('T3_outcomes_engagement',summaries)
    df['engagement_level']=pd.cut(df.active_proportion,[-.001,.25,.5,.75,1],labels=['0–25%','25–50%','50–75%','75–100%'])
    s=df.groupby(['imd_group','engagement_level'],observed=True).agg(n=('success','size'),successes=('success','sum'),success_rate=('success','mean')).reset_index();
    level_masks={str(i):((df.imd_group==r.imd_group)&(df.engagement_level==r.engagement_level)).to_numpy() for i,r in s.iterrows()}
    level_boot=student_descriptive_bootstrap(df,level_masks)
    s['ci_low']=[level_boot[str(i)+'_success'].quantile(.025) for i in s.index]
    s['ci_high']=[level_boot[str(i)+'_success'].quantile(.975) for i in s.index]
    s['ci_method']='student-cluster percentile bootstrap (1000; seed=20261005)'
    table('T3_engagement_levels',s)
    models=[('primary',df,'active_proportion','success',False),('quadratic',df,'active_proportion','success',True),('clicks',df,'log_clicks','success',False),('registration_lag_adjusted',df,'active_proportion','success',False),('consistent_dates',df[~df.inconsistent_withdrawal],'active_proportion','success',False),('conditional_success',df[df.final_result!='Withdrawn'],'active_proportion','success',False),('subsequent_withdrawal',df,'active_proportion','withdrawal',False)]
    records=[];tests=[];main=None
    for name,d,e,o,q in models:
        try:
            m=fit(d,e,o,q,extra='+registration_lag' if name=='registration_lag_adjusted' else '')
            if name=='primary':main=m
            ci=m.conf_int()
            for term in m.params.index:records.append(dict(analysis=name,term=term,estimate=m.params[term],se=m.bse[term],ci_low=ci.loc[term,0],ci_high=ci.loc[term,1],odds_ratio=np.exp(m.params[term]),p=m.pvalues[term],n=len(d),students=d.id_student.nunique(),converged=m.converged))
            idx=[t for t in m.params.index if ':' in t and 'imd_group' in t]
            for label,ii in [('all_groups',idx),('known_IMD',[t for t in idx if 'Missing' not in t])]:
                test=named_wald(m,ii)
                tests.append(dict(analysis=name,contrast=label,chi2=float(test.statistic),df=len(ii),p=float(test.pvalue)))
            print('Statistical model',name,'converged:',m.converged,flush=True)
        except Exception as ex:tests.append(dict(analysis=name,error=repr(ex)))
    table('T4_models',pd.DataFrame(records));tt=pd.DataFrame(tests);mask=tt.analysis.ne('primary') & tt.p.notna();tt.loc[mask,'exploratory_FDR_q']=multipletests(tt.loc[mask,'p'],method='fdr_bh')[1];table('T4_interaction_tests',tt)
    if main is None:raise RuntimeError('Primary association model failed')
    main.save(ROOT/'outputs/models/association.pkl')
    known=['0–30%','30–70%','70–100%'];ranges=df[df.imd_group.isin(known)].groupby('imd_group').active_proportion.quantile([.05,.95]).unstack();low=ranges[.05].max();high=ranges[.95].min()
    table('T5_support_ranges',ranges.reset_index().rename(columns={.05:'p05',.95:'p95'}))
    probs=[];gaps=[];cov=main.cov_params().to_numpy()
    for level in np.unique(np.r_[np.linspace(low,high,21),[x for x in [.25,.5,.75] if low<=x<=high]]):
        support={g:int(((df.imd_group==g)&(abs(df.active_proportion-level)<=.1)).sum()) for g in known}
        if min(support.values())<100:continue
        vals={g:standardize(main,df,g,level) for g in known}
        for g,(p,grad) in vals.items():
            se=np.sqrt(checked_variance(grad@cov@grad));probs.append(dict(group=g,engagement=level,probability=p,ci_low=max(0,p-1.96*se),ci_high=min(1,p+1.96*se),local_n=support[g],standardization_n=len(df)))
        for a,b in [('70–100%','0–30%'),('30–70%','0–30%')]:
            diff=vals[a][0]-vals[b][0];grad=vals[a][1]-vals[b][1];se=np.sqrt(checked_variance(grad@cov@grad));gaps.append(dict(contrast=a+' minus '+b,engagement=level,gap=diff,ci_low=diff-1.96*se,ci_high=diff+1.96*se,n_a=support[a],n_b=support[b]))
    table('T5_adjusted_probabilities',pd.DataFrame(probs));table('T5_adjusted_gaps',pd.DataFrame(gaps))
if __name__=='__main__':run()
