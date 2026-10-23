# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix,roc_curve,precision_recall_curve
from common import *
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.prop_cycle':plt.cycler(color=COLORS),'svg.fonttype':'none'})
def lineband(d,x,y,group,name,title,ylabel):
    fig,ax=plt.subplots(figsize=(7,4))
    for i,(g,a) in enumerate(d.groupby(group,sort=True)):
        ax.plot(a[x],a[y],label=g,color=COLORS[i%4]);ax.fill_between(a[x],a.ci_low,a.ci_high,alpha=.15,color=COLORS[i%4])
    ax.set(xlabel='Proportion of active days by end of week 4',ylabel=ylabel,title=title);ax.legend(fontsize=9);plot_save(name,d,fig)
def run():
    base=pd.read_parquet(ROOT/'data/processed/base.parquet');df=pd.read_parquet(ROOT/'data/processed/week_4.parquet')
    flow=[]
    for w in CFG['weeks']:
        valid=base.date_registration.notna()&(base.date_registration<7*w);e=eligible(base,w)
        flow.append(dict(week=w,raw_enrolments=len(base),raw_students=base.id_student.nunique(),registration_missing=int(base.date_registration.isna().sum()),not_yet_registered=int((base.date_registration>=7*w).sum()),withdrawn_before_cutoff=int((valid&base.date_unregistration.notna()&(base.date_unregistration<7*w)).sum()),eligible_enrolments=int(e.sum()),eligible_students=base.loc[e,'id_student'].nunique()))
    flow=pd.DataFrame(flow);table('T1_flow',flow)
    a=flow[flow.week==4].iloc[0];fig,ax=plt.subplots(figsize=(8,5));ax.axis('off')
    texts=[(0.5,.88,f'Raw enrolments: {len(base):,}\nUnique students: {base.id_student.nunique():,}'),(.5,.58,f'Registered before day 28: {len(base)-a.registration_missing-a.not_yet_registered:,}\nExcluded: {a.registration_missing:,} unknown registration; {a.not_yet_registered:,} later registration'),(.5,.25,f'Week-4 eligible: {a.eligible_enrolments:,} enrolments\n{a.eligible_students:,} unique students\nExcluded recorded withdrawals before day 28: {a.withdrawn_before_cutoff:,}')]
    for x,y,t in texts:ax.text(x,y,t,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',facecolor='#E8F1F8',edgecolor='#0072B2'))
    for y1,y2 in [(.8,.67),(.48,.36)]:ax.annotate('',xy=(.5,y2),xytext=(.5,y1),arrowprops=dict(arrowstyle='->',color='#444444'))
    ax.set_title('Landmark cohort construction: end of week 4');plot_save('cohort_flow',flow,fig)
    counts=df.final_result.value_counts().rename_axis('outcome').reset_index(name='n');fig,ax=plt.subplots(figsize=(7,4));ax.bar(counts.outcome,counts.n,color=COLORS);ax.set(title=f'Final outcomes among week-4 eligible enrolments (N={len(df):,})',ylabel='Enrolments');plot_save('four_outcomes',counts,fig)
    s=pd.read_csv(ROOT/'outputs/tables/T3_outcomes_engagement.csv');fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(s.imd_group,s.success_rate,yerr=[s.success_rate-s.ci_low,s.ci_high-s.success_rate],fmt='o',capsize=4);ax.set(title='Success by residential-area IMD group: week-4 cohort',ylabel='Success proportion (student-cluster bootstrap 95% CI)',xlabel='IMD group; lower bands indicate greater deprivation');ax.set_ylim(0,1);plot_save('success_by_imd',s,fig)
    weekly=[]
    for g,a in df.groupby('imd_group'):
        for j in range(1,5):weekly.append(dict(group=g,week=j,mean_clicks=a[f'clicks_week_{j}'].mean(),mean_active_days=a[f'days_week_{j}'].mean(),n=len(a)))
    weekly=pd.DataFrame(weekly);fig,ax=plt.subplots(figsize=(7,4))
    for g,a in weekly.groupby('group'):ax.plot(a.week,a.mean_active_days,marker='o',label=g)
    ax.set(title='Weekly activity in the fixed week-4 eligible cohort',xlabel='Course week',ylabel='Mean active days');ax.legend();plot_save('weekly_engagement',weekly,fig)
    probs=pd.read_csv(ROOT/'outputs/tables/T5_adjusted_probabilities.csv');lineband(probs,'engagement','probability','group','adjusted_probabilities','Adjusted success associations in supported engagement ranges','Standardised success probability (95% CI)')
    gaps=pd.read_csv(ROOT/'outputs/tables/T5_adjusted_gaps.csv');lineband(gaps,'engagement','gap','contrast','adjusted_gaps','Adjusted success-probability differences between IMD groups','Probability difference (95% CI)')
    metrics=pd.read_csv(ROOT/'outputs/tables/T6_performance.csv')
    for metric in ['ROC_AUC','average_precision']:
        a=metrics[(metrics.model=='Logistic')].copy();fig,ax=plt.subplots(figsize=(7,4))
        for g,b in a.groupby('block'):ax.errorbar(b.week,b[metric],yerr=[b[metric]-b[metric+'_low'],b[metric+'_high']-b[metric]],label='Block '+g,marker='o',capsize=2)
        ax.set(title='Prospective 2014 risk prediction; changing landmark cohorts',xlabel='Observation week',ylabel='ROC-AUC' if metric=='ROC_AUC' else 'Average precision (PR-AUC)');ax.legend();plot_save('windows_'+metric,a,fig)
    selected=json.loads((ROOT/'reports/selected_boosting.json').read_text());pred=pd.read_parquet(ROOT/'outputs/models/predictions.parquet');a=pred[(pred.week==4)&(pred.block=='D')&(pred.split=='2014_test')]
    calibration=[];fig,ax=plt.subplots(figsize=(6,5))
    for name,b in a.groupby('model'):
        if name not in ['Logistic',selected['model'],'Dummy']:continue
        observed,mean=calibration_curve(b.risk,b.probability,n_bins=10,strategy='quantile');ax.plot(mean,observed,marker='o',label=name)
        for x,y in zip(mean,observed):calibration.append(dict(model=name,mean_predicted=x,observed_risk=y,n=len(b)))
    ax.plot([0,1],[0,1],color='gray',linestyle='--');ax.set(title='Week-4 block-D calibration on 2014 students',xlabel='Mean predicted non-success risk',ylabel='Observed non-success proportion');ax.legend();plot_save('calibration',pd.DataFrame(calibration),fig)
    b=a[a.model==selected['model']];cm=confusion_matrix(b.risk,b.probability>=.5);fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Success','Non-success'],yticklabels=['Success','Non-success'],xlabel='Predicted at threshold 0.5',ylabel='Observed',title='Week-4 selected boosting model');plot_save('confusion_matrix',pd.DataFrame([dict(observed=i,predicted=j,n=int(cm[i,j])) for i in range(2) for j in range(2)]),fig)
    gen=pd.read_csv(ROOT/'outputs/tables/T7_generalisation.csv');d=gen[gen.evaluation=='LOMO'];fig,ax=plt.subplots(figsize=(7,4))
    for name,b in d.groupby('model'):ax.plot(b.held_out_module,b.ROC_AUC,marker='o',label=name)
    ax.set(title='Week-4 leave-one-module-out generalisation',xlabel='Held-out module',ylabel='ROC-AUC');ax.legend();plot_save('module_generalisation',d,fig)
    sub=pd.read_csv(ROOT/'outputs/tables/T8_subgroups.csv');d=sub[(sub.week==4)&(sub.block=='D')&(sub.model==selected['model'])];fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(d.imd_group,d.ROC_AUC,yerr=[d.ROC_AUC-d.ROC_AUC_low,d.ROC_AUC_high-d.ROC_AUC],fmt='o',capsize=4);ax.set(title='Week-4 selected boosting performance by IMD',xlabel='Residential-area IMD group',ylabel='ROC-AUC (student-cluster 95% CI)');plot_save('subgroup_performance',d,fig)
if __name__=='__main__':run()
