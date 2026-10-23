# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi
"""Post-audit exploratory adjustment/reference checks; never replace primary tables.
Run from repository root: python src/post_audit_checks.py
"""
import json,hashlib,platform
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import logit
from common import ROOT,validate_inference,checked_variance,named_wald
from association import BASE,standardize

OUT=ROOT/'outputs/post_audit';OUT.mkdir(parents=True,exist_ok=True)

def contrast(model,reference,days):
    a,ga=standardize(model,reference,'70–100%',days/28)
    b,gb=standardize(model,reference,'0–30%',days/28)
    gradient=ga-gb
    se=np.sqrt(checked_variance(gradient@model.cov_params().to_numpy()@gradient))
    return dict(active_days=days,gap=a-b,ci_low=a-b-1.96*se,ci_high=a-b+1.96*se,reference_n=len(reference))

def run():
    path=ROOT/'data/processed/week_4.parquet';d=pd.read_parquet(path)
    for c in ['imd_group','context','gender','age_band','highest_education','disability','region']:d[c]=d[c].fillna('Missing').astype(str)
    specifications={'full':BASE,'context_only':'C(context)','no_education_no_prev_attempts':'C(gender)+C(age_band)+C(disability)+C(region)+studied_credits+C(context)','no_region':'C(gender)+C(age_band)+C(highest_education)+C(disability)+num_of_prev_attempts+studied_credits+C(context)'}
    rows=[];coeff=[];models={}
    for name,rhs in specifications.items():
        m=smf.glm('success ~ C(imd_group)*active_proportion + '+rhs,data=d,family=sm.families.Binomial()).fit(maxiter=100,cov_type='cluster',cov_kwds={'groups':d.id_student.to_numpy()});validate_inference(m);models[name]=m
        for days in [7,14,21]:rows.append(dict(analysis=name,fit_n=len(d),**contrast(m,d,days)))
        ci=m.conf_int()
        for term in m.params.index:coeff.append(dict(analysis=name,term=term,estimate=m.params[term],se=m.bse[term],ci_low=ci.loc[term,0],ci_high=ci.loc[term,1],p=m.pvalues[term],n=len(d),students=d.id_student.nunique(),covariance='student-cluster sandwich'))
        print(name,rows[-1],flush=True)
    primary=models['full'];rows.append(dict(analysis='full_known_IMD_reference',fit_n=len(d),**contrast(primary,d[d.imd_group.ne('Missing')],21)))
    # Explicit region whitelist; report excluded categories rather than infer nationality.
    english=['East Anglian Region','East Midlands Region','London Region','North Region','North Western Region','South East Region','South Region','South West Region','West Midlands Region','Yorkshire Region']
    eng=d[d.region.isin(english)].copy();assert len(eng)>0
    m=smf.glm('success ~ C(imd_group)*active_proportion + '+BASE,data=eng,family=sm.families.Binomial()).fit(maxiter=100,cov_type='cluster',cov_kwds={'groups':eng.id_student.to_numpy()});validate_inference(m)
    rows.append(dict(analysis='english_regions_only',fit_n=len(eng),**contrast(m,eng,21)))
    ci=m.conf_int()
    for term in m.params.index:coeff.append(dict(analysis='english_regions_only',term=term,estimate=m.params[term],se=m.bse[term],ci_low=ci.loc[term,0],ci_high=ci.loc[term,1],p=m.pvalues[term],n=len(eng),students=eng.id_student.nunique(),covariance='student-cluster sandwich'))
    terms=[t for t in m.params.index if ':' in t and 'imd_group' in t and 'Missing' not in t];wald=named_wald(m,terms)
    pd.DataFrame(rows).to_csv(OUT/'adjustment_contrasts.csv',index=False);pd.DataFrame(coeff).to_csv(OUT/'adjustment_coefficients.csv',index=False)
    # Calibration intercept and slope on archived, calibrated 2014 probabilities;
    # these are descriptive evaluation fits, never applied to update test scores.
    pred=pd.read_parquet(ROOT/'outputs/models/predictions.parquet');print('Prediction columns:',pred.columns.tolist(),flush=True)
    cal=[]
    for block in ['B','C','D']:
        a=pred[(pred.week==4)&pred.block.eq(block)&pred.model.eq('Logistic')&pred.split.eq('2014_test')].copy()
        assert len(a)==15722 and a.probability.between(0,1,inclusive='neither').all()
        z=logit(a.probability.to_numpy());X=sm.add_constant(z)
        fit=sm.GLM(a.risk.to_numpy(),X,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':a.id_student.to_numpy()})
        intercept=sm.GLM(a.risk.to_numpy(),np.ones((len(a),1)),offset=z,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':a.id_student.to_numpy()})
        ci=fit.conf_int()
        cal.append(dict(block=block,n=len(a),calibration_in_large=float(intercept.params[0]),joint_intercept=float(fit.params[0]),slope=float(fit.params[1]),slope_ci_low=float(ci[1,0]),slope_ci_high=float(ci[1,1]),Brier=float(np.mean((a.probability-a.risk)**2))))
    pd.DataFrame(cal).to_csv(OUT/'test_calibration.csv',index=False)
    print('Calibration',cal,flush=True)
    known=d[d.imd_group.ne('Missing')]
    summary=dict(python=platform.python_version(),analysis_stage='exploratory additions after Claude-assisted internal computational audit, 22 October 2026',cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),english_regions=list(sorted(eng.region.unique())),english_n=len(eng),english_known_IMD_wald_p=float(wald.pvalue),missing_IMD_by_region=d[d.imd_group.eq('Missing')].region.value_counts().to_dict(),known_IMD_nonenglish_share=float(known.region.isin(['Scotland','Wales','Ireland']).mean()),share_at_least_21_active_days=d.assign(atleast=d.active_days.ge(21)).groupby('imd_group').atleast.mean().to_dict(),raw_date_outcome_disagreements=None)
    b=pd.read_parquet(ROOT/'data/processed/base.parquet');summary['raw_date_outcome_disagreements']=int((b.date_unregistration.ge(28)&b.final_result.ne('Withdrawn')).sum())
    reg0=b[b.date_registration.notna()&b.date_registration.le(0)]
    rates=reg0.groupby('imd_group').success.mean()
    summary['registered_by_day0']=dict(n=len(reg0),unadjusted_least_minus_most_gap=float(rates['70–100%']-rates['0–30%']))
    summary['model_formulas']={key:'success ~ C(imd_group)*active_proportion + '+rhs for key,rhs in specifications.items()}
    (OUT/'checks.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':run()
