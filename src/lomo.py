# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

"""Strict holdout-specific tuning/calibration: no global validation selection."""
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,f1_score
from common import *
from ml import pipeline,split2013,calibrated,measure,bootstrap
def run():
    df=pd.read_parquet(ROOT/'data/processed/week_4.parquet');rows=[];pred=[];split=[];tuning=[]
    for module in sorted(df.code_module.unique()):
        test=df[df.code_module==module];train=df[(df.code_module!=module)&~df.id_student.isin(test.id_student)];f,v,c=split2013(train)
        for label,a in [('fit',f),('validation',v),('calibration',c),('test',test)]:
            s=a[KEY].copy();s['week']=4;s['evaluation']='LOMO_'+module;s['split']=label;split.append(s)
        for name in ['Logistic','XGBoost','LightGBM']:
            best=-np.inf;chosen=None
            for param in ([.3,1] if name=='Logistic' else [2,4]):
                m=pipeline(f,'D',name,param);m.fit(f,f.risk);auc=roc_auc_score(v.risk,m.predict_proba(v)[:,1]);tuning.append(dict(module=module,model=name,param=param,validation_AUC=auc))
                if auc>best:best=auc;chosen=param
            m=pipeline(f,'D',name,chosen);fitall=pd.concat([f,v]);m.fit(fitall,fitall.risk)
            pc=m.predict_proba(c)[:,1];z=np.log(np.clip(pc,1e-6,1-1e-6)/np.clip(1-pc,1e-6,1))[:,None];cal=LogisticRegression(C=1e3,max_iter=1000).fit(z,c.risk);cp=calibrated(pc,cal);threshold=float(max(np.linspace(.1,.9,81),key=lambda t:f1_score(c.risk,cp>=t)))
            p=calibrated(m.predict_proba(test)[:,1],cal);y=test.risk.to_numpy();b=bootstrap(y,p,test.id_student.to_numpy());r=dict(week=4,block='D',model=name,evaluation='LOMO',held_out_module=module,n=len(test),train_n=len(train),chosen_param=chosen,validation_AUC=best,overlap_excluded=int(((df.code_module!=module)&df.id_student.isin(test.id_student)).sum()),prevalence=y.mean(),**measure(y,p))
            for metric in b:r[metric+'_low'],r[metric+'_high']=b[metric].quantile([.025,.975])
            rows.append(r);out=test[KEY+['imd_group','presentation_type','risk']].copy();out['probability']=p;out['week']=4;out['block']='D';out['model']=name;out['split']='LOMO_'+module;out['threshold']=threshold;pred.append(out)
            print('Strict LOMO',module,name,flush=True)
    d=pd.read_csv(ROOT/'outputs/tables/T7_generalisation.csv');table('T7_generalisation',pd.concat([d[d.evaluation!='LOMO'],pd.DataFrame(rows)],ignore_index=True))
    d=pd.read_parquet(ROOT/'outputs/models/predictions.parquet');pd.concat([d[d.split=='2014_test'],pd.concat(pred)],ignore_index=True).to_parquet(ROOT/'outputs/models/predictions.parquet',index=False)
    d=pd.read_csv(ROOT/'reports/split_assignments.csv');pd.concat([d[d.evaluation=='temporal'],pd.concat(split)],ignore_index=True).to_csv(ROOT/'reports/split_assignments.csv',index=False);pd.DataFrame(tuning).to_csv(ROOT/'reports/lomo_tuning.csv',index=False)
if __name__=='__main__':run()
