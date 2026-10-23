# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

from common import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
BASE=ROOT/'outputs/first_project';FOLDER=BASE/'figures';FOLDER.mkdir(parents=True,exist_ok=True)
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
plt.rcParams.update({'font.size':10,'font.family':'Arial','svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def save(name,data,fig):
    data.to_csv(FOLDER/f'{name}.csv',index=False);fig.savefig(FOLDER/f'{name}.png',dpi=300,bbox_inches='tight');fig.savefig(FOLDER/f'{name}.svg',bbox_inches='tight');Image.open(FOLDER/f'{name}.png').convert('RGB').save(FOLDER/f'{name}.tif',dpi=(300,300),compression='tiff_lzw');plt.close(fig)

def module_figure():
    """Plot the post-analysis module fits, without refitting or inventing estimates."""
    source=ROOT/'outputs/manuscript_checks/module_OR.csv'
    if not source.exists():
        raise FileNotFoundError('Run src/manuscript_checks.py before generating the module figure.')
    d=pd.read_csv(source)
    groups=['0–30%','30–70%','70–100%'];modules=sorted(d.module.unique())
    assert len(modules)==7 and len(d)==21 and not d.duplicated(['module','group']).any()
    assert np.isfinite(d[['OR','low','high']]).all().all()
    assert ((d.low>0)&(d.low<=d.OR)&(d.OR<=d.high)).all()
    fig,ax=plt.subplots(figsize=(7.2,5.4),layout='constrained')
    y=np.arange(len(modules))
    for g,color,marker,offset in zip(groups,COLORS,['o','s','^'],[-.23,0,.23]):
        a=d[d.group.eq(g)].set_index('module').loc[modules]
        ax.errorbar(a.OR,y+offset,xerr=[a.OR-a.low,a.high-a.OR],
            fmt=marker,color=color,capsize=3,markersize=5,label=g,linewidth=1.3)
    ax.axvline(1,color='gray',linestyle='--',linewidth=1)
    ax.set_xscale('log');ax.set_xlim(.95,4.6)
    ax.set_xticks([1,1.5,2,3,4],labels=['1','1.5','2','3','4'])
    ax.minorticks_off();ax.set_yticks(y,labels=modules);ax.invert_yaxis()
    ax.set(xlabel='Adjusted success odds ratio per seven additional active days (log scale)',
        ylabel='Course module',title='Engagement–success associations across seven modules')
    ax.legend(title='IMD group',loc='upper right',frameon=False)
    save('module_engagement_success_OR',d,fig)
def run():
    d=pd.read_csv(BASE/'tables/S1_group_descriptives.csv');fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(d.imd_group,d.success_rate*100,yerr=[(d.success_rate-d.success_ci_low)*100,(d.success_ci_high-d.success_rate)*100],fmt='o',capsize=4);ax.set(xlabel='Residential-area IMD group (lower bands = greater deprivation)',ylabel='Observed success (%)',title='Week-4 cohort success: student-cluster 95% intervals');ax.set_ylim(0,100);save('observed_success',d,fig)
    d=pd.read_csv(BASE/'tables/S3_adjusted_engagement_associations.csv');a=d[(d.analysis=='primary_success')&(d.imd_group!='Missing')];fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(a.OR_per_7_active_days,a.imd_group,xerr=[a.OR_per_7_active_days-a.OR_ci_low,a.OR_ci_high-a.OR_per_7_active_days],fmt='o',capsize=4);ax.axvline(1,color='gray',linestyle='--');ax.set(xlabel='Adjusted success odds ratio per seven additional active days',ylabel='IMD group',title='Engagement–success associations: student-cluster 95% intervals');save('adjusted_engagement_success_OR',a,fig)
    d=pd.read_csv(BASE/'tables/S6_adjusted_success_probabilities.csv');fig,ax=plt.subplots(figsize=(7,4))
    for color,(group,a) in zip(COLORS,d.groupby('group')):ax.errorbar(a.engagement*28,a.probability*100,yerr=[(a.probability-a.ci_low)*100,(a.ci_high-a.probability)*100],marker='o',label=group,color=color,capsize=4)
    ax.set(xlabel='Active days in the first 28 days',ylabel='Standardised success (%)',title='Adjusted success associations at supported engagement levels');ax.set_xticks([7,14,21]);ax.legend(title='IMD');save('adjusted_success_probabilities',d,fig)
    d=pd.read_csv(BASE/'tables/S7_adjusted_success_gaps.csv');fig,ax=plt.subplots(figsize=(7,4))
    for color,(group,a) in zip(COLORS,d.groupby('contrast')):ax.errorbar(a.engagement*28,a.gap*100,yerr=[(a.gap-a.ci_low)*100,(a.ci_high-a.gap)*100],marker='o',label=group,color=color,capsize=4)
    ax.axhline(0,color='gray',linestyle='--');ax.set(xlabel='Active days in the first 28 days',ylabel='Success contrast (percentage points)',title='Covariate-standardised success contrasts');ax.set_xticks([7,14,21]);ax.legend(fontsize=9);save('adjusted_success_gaps',d,fig)
    d=pd.read_csv(BASE/'tables/S3_adjusted_engagement_associations.csv');a=d[(d.analysis=='subsequent_withdrawal')&(d.imd_group!='Missing')];fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(a.OR_per_7_active_days,a.imd_group,xerr=[a.OR_per_7_active_days-a.OR_ci_low,a.OR_ci_high-a.OR_per_7_active_days],fmt='o',capsize=4);ax.axvline(1,color='gray',linestyle='--');ax.set(xlabel='Adjusted withdrawal odds ratio per seven additional active days',ylabel='IMD group',title='Exploratory outcome-specific engagement–withdrawal associations');save('exploratory_withdrawal_OR',a,fig)
if __name__=='__main__':
    run()
    if (ROOT/'outputs/manuscript_checks/module_OR.csv').exists():module_figure()

