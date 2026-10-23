# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

from common import *
import json,joblib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from common import *
from features import refresh_click_features
from sklearn.model_selection import train_test_split


def stratified_students(df, n):
    # One reproducible enrolment per student, stratified by IMD and outcome.
    students=df.sample(frac=1,random_state=CFG['seed']).drop_duplicates('id_student')
    if len(students)<1000:
        raise RuntimeError('At least 1000 distinct evaluation students are required for SHAP')
    n=min(max(1000,n),len(students))
    if n==len(students):return students
    strata=students.imd_group.astype(str)+'_'+students.success.astype(str)
    sample,_=train_test_split(students,train_size=n,stratify=strata,random_state=CFG['seed'])
    return sample


def coherent_engagement(frame, proportion, reference):
    """Use observed neighbouring engagement profiles within module and IMD.

    This is a joint engagement-profile ALE path, not a one-variable causal effect.
    Donors preserve discrete weekly days, resource counts and recency support.
    """
    out=frame.copy()
    columns=['active_days','active_proportion','active_weeks','resources_accessed',
             'activity_diversity','days_since_activity','total_clicks']
    columns += [f'{kind}_week_{j}' for kind in ['days','clicks'] for j in range(1,5)]
    for (module,group), block in frame.groupby(['code_module','imd_group'],observed=True):
        pool=reference[(reference.code_module==module)&(reference.imd_group==group)]
        distance=(pool.active_proportion-proportion).abs()
        donors=pool.loc[distance==distance.min()].sort_values('id_student')
        # Match intensity and weekly activity shape as closely as support permits.
        match=['clicks_per_active_day','activity_diversity']+[f'days_week_{j}' for j in range(1,5)]
        original=block[match].to_numpy(dtype=float,copy=True)
        candidate=donors[match].to_numpy(dtype=float,copy=True)
        original[:,0]=np.log1p(original[:,0]);candidate[:,0]=np.log1p(candidate[:,0])
        original[:,2:]/=np.maximum(block.active_days.to_numpy()[:,None],1)
        candidate[:,2:]/=np.maximum(donors.active_days.to_numpy()[:,None],1)
        scale=np.maximum(candidate.std(axis=0),1)
        indices=np.square((original[:,None,:]-candidate[None,:,:])/scale).sum(axis=2).argmin(axis=1)
        out.loc[block.index,columns]=donors.iloc[indices][columns].to_numpy()
    refresh_click_features(out,4)
    if not np.allclose(out.active_days,out[[f'days_week_{j}' for j in range(1,5)]].sum(axis=1)):
        raise RuntimeError('Inconsistent weekly-day totals in ALE profiles')
    if not np.allclose(out.total_clicks,out[[f'clicks_week_{j}' for j in range(1,5)]].sum(axis=1)):
        raise RuntimeError('Inconsistent weekly-click totals in ALE profiles')
    return out

def run():
    selected=json.loads((ROOT/'reports/selected_boosting.json').read_text());bundle=joblib.load(ROOT/selected['path']);pipe=bundle['pipeline']
    df=pd.read_parquet(ROOT/'data/processed/week_4.parquet');test=stratified_students(df[df.year==2014],CFG['shap_sample'])
    x=pipe.named_steps['preprocess'].transform(test);names=np.asarray(pipe.named_steps['preprocess'].get_feature_names_out(),dtype=str);explainer=shap.TreeExplainer(pipe.named_steps['model'],feature_perturbation='tree_path_dependent',model_output='raw');values=explainer.shap_values(x)
    if isinstance(values,list):values=values[1]
    if np.ndim(values)==3:values=values[:,:,1]
    assert values.shape==x.shape
    importance=pd.DataFrame({'feature':names,'mean_absolute_SHAP':abs(values).mean(axis=0)}).sort_values('mean_absolute_SHAP',ascending=False);table('T8_SHAP_importance',importance)
    data=pd.DataFrame(values,columns=names);data.insert(0,'id_student',test.id_student.to_numpy());data.to_csv(ROOT/'outputs/figures/shap_sample_values.csv',index=False)
    shap.summary_plot(values,x,feature_names=names,max_display=20,show=False);plot_save('shap_global',importance,plt.gcf())
    for kind,mask in [('imd',['imd_group']),('engagement',['active_','click','week_','engagement','recent_change','days_since','resources','diversity'])]:
        idx=[i for i,n in enumerate(names) if any(k in n for k in mask)];a=importance[importance.feature.isin(names[idx])].head(15).sort_values('mean_absolute_SHAP');fig,ax=plt.subplots(figsize=(8,5));ax.barh(a.feature,a.mean_absolute_SHAP);ax.set(title=f'SHAP {kind} explanations: validation-selected {selected["model"]}',xlabel='Mean absolute SHAP value (raw model log-odds)');plot_save('shap_'+kind,a,fig)
    table('T8_SHAP_sampling',test.groupby(['imd_group','success'],observed=True).agg(students=('id_student','nunique')).reset_index())
    interaction_note=''
    try:
        interaction=explainer.shap_interaction_values(x)
        if isinstance(interaction,list):interaction=interaction[1]
        if np.ndim(interaction)==4:interaction=interaction[:,:,:,1]
        if interaction.shape!=(len(test),x.shape[1],x.shape[1]):
            raise ValueError('Unexpected TreeSHAP interaction dimensions')
        if not np.isfinite(interaction).all() or not np.allclose(interaction,interaction.transpose(0,2,1),atol=1e-4,rtol=1e-5):
            raise ValueError('TreeSHAP interactions must be finite and symmetric')
        if not np.allclose(interaction.sum(axis=2),values,atol=1e-4,rtol=1e-5):
            raise ValueError('TreeSHAP interaction sums do not reproduce SHAP values')
        (ROOT/'reports/shap_interaction_checks.json').write_text(json.dumps(dict(sample_n=len(test),unique_students=int(test.id_student.nunique()),seed=CFG['seed'],stratification=['imd_group','success'],finite=True,max_symmetry_error=float(np.max(abs(interaction-interaction.transpose(0,2,1)))),max_sum_to_shap_error=float(np.max(abs(interaction.sum(axis=2)-values)))),indent=2),encoding='utf-8')
        imd=[i for i,n in enumerate(names) if 'imd_group' in n];eng=[i for i,n in enumerate(names) if n.startswith('num__') and any(k in n for k in ['active','click','week','engagement','recent','resources','diversity','days_since'])]
        rows=[dict(feature=names[j],mean_absolute_IMD_interaction=float(abs(interaction[:,imd,j].sum(axis=1)).mean()),sample_n=len(interaction)) for j in eng]
        a=pd.DataFrame(rows).sort_values('mean_absolute_IMD_interaction').tail(15);table('T8_SHAP_IMD_interactions',pd.DataFrame(rows))
        np.savez_compressed(ROOT/'outputs/models/shap_interactions.npz',values=interaction,features=names,students=test.id_student.to_numpy())
        fig,ax=plt.subplots(figsize=(8,5));ax.barh(a.feature,a.mean_absolute_IMD_interaction);ax.set(title='Verified TreeSHAP IMD × engagement interactions',xlabel='Mean absolute grouped interaction (raw log-odds)');plot_save('shap_imd_interactions',a,fig)
        interaction_note='TreeSHAP interaction values were computed on '+str(len(test))+' distinct, IMD/outcome-stratified test students and verified for symmetry and sum-to-SHAP consistency. Encoded IMD interaction entries are summed before taking absolute means. These describe the fitted tree, not the confirmatory regression or a causal interaction.'
    except Exception as ex:
        raise RuntimeError('TreeSHAP interaction computation or symmetry/sum validation failed') from ex
    # Joint observed-profile ALE on calibrated risk scale.
    from ml import calibrated
    edges=np.unique(np.quantile(test.active_proportion,np.linspace(0,1,11)));rows=[];cumulative=0
    for lo,hi in zip(edges[:-1],edges[1:]):
        subset=test[(test.active_proportion>=lo)&((test.active_proportion<hi) if hi<edges[-1] else (test.active_proportion<=hi))].copy()
        if len(subset)<10:continue
        left=coherent_engagement(subset,lo,df[df.year==2014]);right=coherent_engagement(subset,hi,df[df.year==2014])
        delta=(calibrated(pipe.predict_proba(right)[:,1],bundle['calibrator'])-calibrated(pipe.predict_proba(left)[:,1],bundle['calibrator'])).mean();cumulative+=delta;rows.append(dict(midpoint=(lo+hi)/2,ale=cumulative,n=len(subset),lower=lo,upper=hi,realized_lower=left.active_proportion.mean(),realized_upper=right.active_proportion.mean()))
    a=pd.DataFrame(rows);a['ale']=a.ale-np.average(a.ale,weights=a.n);fig,ax=plt.subplots(figsize=(7,4));ax.plot(a.midpoint,a.ale,marker='o');ax.set(title='Joint engagement-profile ALE; descriptive model explanation',xlabel='Active-day proportion',ylabel='Centred change in predicted non-success probability');plot_save('ale_engagement',a,fig)
    (ROOT/'reports/explanation_notes.md').write_text('SHAP values explain the uncalibrated tree model on its raw output scale. Calibration is separate. Sampling is reproducible. '+interaction_note+' ALE uses neighbouring observed engagement profiles within module and IMD, updating active days, weekly days, clicks, recency and resource summaries jointly and recalculating log clicks, intensity and trends. This joint profile curve is not an isolated active-proportion effect or a causal effect.\n',encoding='utf-8')
if __name__=='__main__':run()
