# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

"""Focused statistical exports and factual interpretation for proposal one."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from common import ROOT,CFG,KEY,student_descriptive_bootstrap,named_wald,checked_variance
from association import fit
OUT=ROOT/'outputs/first_project'
GROUPS=['0–30%','30–70%','70–100%','Missing']
def export(name,d):
    folder=OUT/'tables';folder.mkdir(parents=True,exist_ok=True)
    d.to_csv(folder/f'{name}.csv',index=False)
    (folder/f'{name}.md').write_text(d.to_markdown(index=False),encoding='utf-8')
def linear(m,r):
    beta=float(r@m.params.to_numpy());se=float(np.sqrt(checked_variance(r@m.cov_params().to_numpy()@r)))
    return beta,se,float(2*norm.sf(abs(beta/se))) if se else np.nan
def slope_vector(m,g):
    r=np.zeros(len(m.params));r[list(m.params.index).index('active_proportion')]=1
    term=f'C(imd_group)[T.{g}]:active_proportion'
    if term in m.params.index:r[list(m.params.index).index(term)]=1
    return r
def run():
    df=pd.read_parquet(ROOT/'data/processed/week_4.parquet')
    summary=[]
    masks={g:df.imd_group.eq(g).to_numpy() for g in GROUPS}
    boot=student_descriptive_bootstrap(df,masks)
    for g,ix in masks.items():
        a=df[ix];lo,hi=boot[g+'_success'].quantile([.025,.975]);el,eh=boot[g+'_active'].quantile([.025,.975])
        summary.append(dict(imd_group=g,enrolments=len(a),unique_students=a.id_student.nunique(),successes=int(a.success.sum()),success_rate=a.success.mean(),success_ci_low=lo,success_ci_high=hi,mean_active_proportion=a.active_proportion.mean(),engagement_ci_low=el,engagement_ci_high=eh,mean_active_days=a.active_days.mean(),median_clicks=a.total_clicks.median(),zero_clicks=int(a.total_clicks.eq(0).sum())))
    summary=pd.DataFrame(summary);export('S1_group_descriptives',summary)
    observed=[]
    for a,b in [('70–100%','0–30%'),('30–70%','0–30%')]:
        for field,col in [('success_rate','success'),('mean_active_proportion','active')]:
            difference=summary.set_index('imd_group').loc[a,field]-summary.set_index('imd_group').loc[b,field];lo,hi=(boot[a+'_'+col]-boot[b+'_'+col]).quantile([.025,.975]);observed.append(dict(contrast=a+' minus '+b,measure=field,difference=difference,ci_low=lo,ci_high=hi))
    export('S2_observed_group_gaps',pd.DataFrame(observed))
    slopes=[];contrasts=[];modelinfo=[]
    for name,d,outcome in [('primary_success',df,'success'),('conditional_success',df[df.final_result!='Withdrawn'],'success'),('subsequent_withdrawal',df,'withdrawal')]:
        m=fit(d,outcome=outcome)
        if name=='primary_success':
            reference=pd.read_csv(ROOT/'outputs/tables/T4_models.csv');reference=reference[reference.analysis=='primary'].set_index('term');assert np.allclose(m.params,reference.loc[m.params.index,'estimate'],atol=1e-10)
        vectors={g:slope_vector(m,g) for g in GROUPS}
        for g,r in vectors.items():
            beta,se,p=linear(m,r*.25);slopes.append(dict(analysis=name,imd_group=g,positive_outcome=outcome,log_odds_difference=beta,cluster_se=se,OR_per_7_active_days=np.exp(beta),OR_ci_low=np.exp(beta-1.96*se),OR_ci_high=np.exp(beta+1.96*se),p=p,enrolments=int(d.imd_group.eq(g).sum()),unique_students=d.loc[d.imd_group.eq(g),'id_student'].nunique()))
        start=len(contrasts)
        for a,b in [('30–70%','0–30%'),('70–100%','0–30%'),('70–100%','30–70%')]:
            beta,se,p=linear(m,(vectors[a]-vectors[b])*.25);contrasts.append(dict(analysis=name,contrast=a+' relative to '+b,ratio_of_engagement_ORs=np.exp(beta),ratio_ci_low=np.exp(beta-1.96*se),ratio_ci_high=np.exp(beta+1.96*se),p=p))
        q=multipletests([r['p'] for r in contrasts[start:]],method='fdr_bh')[1]
        for r,val in zip(contrasts[start:],q):r['post_analysis_FDR_q']=val
        known=[t for t in m.params.index if ':' in t and 'imd_group' in t and 'Missing' not in t];test=named_wald(m,known)
        modelinfo.append(dict(analysis=name,n=len(d),unique_students=d.id_student.nunique(),converged=m.converged,interaction_chi2=float(test.statistic),interaction_df=len(known),interaction_p=float(test.pvalue)))
        print(name,'slopes derived; n=',len(d),flush=True)
    slopes=pd.DataFrame(slopes);contrasts=pd.DataFrame(contrasts);export('S3_adjusted_engagement_associations',slopes);export('S4_engagement_OR_comparisons',contrasts);export('S5_model_information',pd.DataFrame(modelinfo))
    for source,target in [('T5_adjusted_probabilities','S6_adjusted_success_probabilities'),('T5_adjusted_gaps','S7_adjusted_success_gaps')]:
        a=pd.read_csv(ROOT/f'outputs/tables/{source}.csv');a=a[a.engagement.apply(lambda x:any(np.isclose(x,z) for z in [.25,.5,.75]))].copy();export(target,a)
    t=pd.read_csv(ROOT/'outputs/tables/T4_interaction_tests.csv');export('S8_sensitivity_interactions',t[t.contrast=='known_IMD'])
    export('S9_original_IMD_descriptives',pd.read_csv(ROOT/'outputs/tables/T3_original_IMD_bands.csv'))
    data=[]
    for p in sorted((OUT/'tables').glob('S*.csv')):
        d=pd.read_csv(p);data.append(dict(name=p.stem.split('_')[0],title=p.stem,columns=list(d),values=d.astype(object).where(pd.notna(d),None).values.tolist()))
    (OUT/'tables/workbook_data.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    (ROOT/'reports/first_project_manifest.json').write_text(json.dumps(dict(reporting_extension=True,primary_model_unchanged=True,window_week=4,bootstrap=1000,seed=CFG['seed'],engagement_OR_increment=.25,plan_sha256=hashlib.sha256((ROOT/'analysis_plan.md').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':run()
