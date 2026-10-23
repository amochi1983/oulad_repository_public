# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

import warnings,json,joblib
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,precision_score,recall_score,balanced_accuracy_score,brier_score_loss,confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from common import *
CAT=['gender','age_band','highest_education','disability','region','code_module','presentation_type']
NUM=['num_of_prev_attempts','studied_credits']
CUM=['log_clicks','active_days','active_weeks','active_proportion','clicks_per_active_day','resources_accessed','activity_diversity','days_since_activity','precourse_clicks','precourse_active_days']
METRICS=['ROC_AUC','average_precision','F1','precision','recall','balanced_accuracy','Brier','FPR','FNR','calibration_bias']
def columns(df,block):
    c=CAT+([] if block=='A' else ['imd_group']);n=NUM.copy()
    if block in ['C','D']:n+=CUM
    if block=='D':n += [x for x in df if x.startswith(('clicks_week_','days_week_')) or x in ['engagement_slope','engagement_sd','recent_change']]
    return c,n
def estimator(name,param):
    seed=CFG['seed'];trees=CFG['trees'];threads=CFG['threads']
    if name=='Dummy':return DummyClassifier(strategy='prior')
    if name=='Logistic':return LogisticRegression(C=param,max_iter=1500,random_state=seed)
    if name=='RandomForest':return RandomForestClassifier(n_estimators=trees,max_depth=param,min_samples_leaf=10,n_jobs=threads,random_state=seed)
    if name=='XGBoost':return XGBClassifier(n_estimators=trees,max_depth=param,learning_rate=.06,subsample=.85,colsample_bytree=.85,n_jobs=threads,random_state=seed,eval_metric='logloss')
    return LGBMClassifier(n_estimators=trees,max_depth=param,num_leaves=15,learning_rate=.06,verbosity=-1,n_jobs=threads,random_state=seed)
def pipeline(df,block,name,param):
    cat,num=columns(df,block)
    prep=ColumnTransformer([('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat),('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num)],sparse_threshold=0)
    return Pipeline([('preprocess',prep),('model',estimator(name,param))])
def measure(y,p,threshold=.5,alert=None):
    pred=(p>=threshold).astype(int) if alert is None else alert
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(ROC_AUC=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,average_precision=average_precision_score(y,p) if sum(y)>0 else np.nan,F1=f1_score(y,pred,zero_division=0),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),balanced_accuracy=balanced_accuracy_score(y,pred),Brier=brier_score_loss(y,p),FPR=fp/(tn+fp) if tn+fp else np.nan,FNR=fn/(tp+fn) if tp+fn else np.nan,calibration_bias=float(np.mean(p)-np.mean(y)))
def bootstrap(y,p,ids,threshold=.5,alert=None):
    groups=pd.factorize(ids)[0];unique=np.unique(groups);rng=np.random.default_rng(CFG['seed']);out=[]
    order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order];gs=groups[order]
    ends=np.r_[np.flatnonzero(np.diff(ps)!=0),len(ps)-1]
    original_flag=ps>=threshold if alert is None else np.asarray(alert)[order].astype(bool)
    for _ in range(CFG['bootstrap']):
        weights=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[gs]
        positive=weights*ys;negative=weights*(1-ys);P=positive.sum();N=negative.sum();total=P+N
        tp=np.cumsum(positive)[ends];fp=np.cumsum(negative)[ends]
        auc=np.trapezoid(np.r_[0,tp/P],np.r_[0,fp/N]) if P and N else np.nan
        ap=np.sum(np.diff(np.r_[0,tp/P])*np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=tp+fp>0)) if P else np.nan
        flag=original_flag;TP=positive[flag].sum();FP=negative[flag].sum();FN=P-TP
        prec=TP/(TP+FP) if TP+FP else 0;rec=TP/P if P else 0
        out.append(dict(ROC_AUC=auc,average_precision=ap,F1=2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else 0,precision=prec,recall=rec,balanced_accuracy=(rec+(1-FP/N))/2 if P and N else np.nan,Brier=np.sum(weights*(ps-ys)**2)/total,FPR=FP/N if N else np.nan,FNR=FN/P if P else np.nan,calibration_bias=np.sum(weights*(ps-ys))/total))
    return pd.DataFrame(out)
def split2013(train):
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=CFG['seed']).split(train,groups=train.id_student))
    rem=train.iloc[a];fitidx,valididx=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=CFG['seed']+1).split(rem,groups=rem.id_student))
    return rem.iloc[fitidx],rem.iloc[valididx],train.iloc[b]
def calibrated(p,cal):
    # Platt calibration learned on untouched 2013 calibration students.
    z=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))[:,None]
    return cal.predict_proba(z)[:,1]
def run(smoke=False):
    rows=[];subrows=[];general=[];predictions=[];splits=[];tuning=[];policyrows=[];paired=[];best=None;bestval=-np.inf
    weeks=[4] if smoke else CFG['weeks']
    names=['Dummy','Logistic'] if smoke else ['Dummy','Logistic','RandomForest','XGBoost','LightGBM']
    for w in weeks:
        df=pd.read_parquet(ROOT/f'data/processed/week_{w}.parquet')
        test=df[df.year==2014].copy();train=df[(df.year==2013)&~df.id_student.isin(test.id_student)].copy()
        f,v,c=split2013(train)
        for label,a in [('fit',f),('validation',v),('calibration',c),('test',test)]:
            s=a[KEY].copy();s['week']=w;s['evaluation']='temporal';s['split']=label;splits.append(s)
        general.append(dict(week=w,evaluation='temporal_composition',train_n=len(train),test_n=len(test),overlap_excluded=int(((df.year==2013)&df.id_student.isin(test.id_student)).sum()),train_modules=','.join(sorted(train.code_module.unique())),test_modules=','.join(sorted(test.code_module.unique())),train_prevalence=train.risk.mean(),test_prevalence=test.risk.mean()))
        draw_cache={}
        for block in (['B','D'] if smoke else ['A','B','C','D']):
            for name in names:
                params=[1] if name=='Dummy' else ([.3,1] if name=='Logistic' else ([6,12] if name=='RandomForest' else [2,4]))
                selected=None;score=-np.inf;chosen=None
                for param in params:
                    m=pipeline(f,block,name,param);m.fit(f,f.risk);auc=roc_auc_score(v.risk,m.predict_proba(v)[:,1]);tuning.append(dict(week=w,block=block,model=name,param=param,validation_AUC=auc))
                    if auc>score:score=auc;selected=m;chosen=param
                fitall=pd.concat([f,v]);m=pipeline(f,block,name,chosen);m.fit(fitall,fitall.risk)
                pc=m.predict_proba(c)[:,1];z=np.log(np.clip(pc,1e-6,1-1e-6)/np.clip(1-pc,1e-6,1))[:,None]
                cal=LogisticRegression(C=1e3,max_iter=1000).fit(z,c.risk)
                cp=calibrated(pc,cal);thresholds=np.linspace(.1,.9,81);threshold=float(max(thresholds,key=lambda t:f1_score(c.risk,cp>=t,zero_division=0)))
                p=calibrated(m.predict_proba(test)[:,1],cal);y=test.risk.to_numpy();ids=test.id_student.to_numpy()
                met=measure(y,p);boot=bootstrap(y,p,ids);draw_cache[(block,name)]=boot
                row=dict(week=w,block=block,model=name,n=len(test),students=test.id_student.nunique(),prevalence=y.mean(),validation_AUC=score,chosen_param=chosen,validation_threshold=threshold,**met)
                for metric in METRICS:row[metric+'_low'],row[metric+'_high']=boot[metric].quantile([.025,.975])
                rows.append(row)
                for typ in ['B','J']:
                    ix=test.presentation_type.eq(typ).to_numpy();general.append(dict(week=w,block=block,model=name,evaluation='temporal_'+typ,n=int(ix.sum()),prevalence=y[ix].mean(),**measure(y[ix],p[ix])))
                for g in sorted(test.imd_group.unique()):
                    ix=test.imd_group.eq(g).to_numpy();b=bootstrap(y[ix],p[ix],ids[ix]);r=dict(week=w,block=block,model=name,imd_group=g,n=int(ix.sum()),students=len(np.unique(ids[ix])),prevalence=y[ix].mean(),unstable=ix.sum()<200,**measure(y[ix],p[ix]))
                    for metric in METRICS:r[metric+'_low'],r[metric+'_high']=b[metric].quantile([.025,.975])
                    subrows.append(r)
                for policy in ['0.5','validation_F1','top10','top20']:
                    alert=None;t=.5
                    if policy=='validation_F1':t=threshold
                    if policy.startswith('top'):
                        alert=np.zeros(len(y),dtype=int);count=int(np.ceil(len(y)*int(policy[3:])/100));alert[np.argsort(-p,kind='stable')[:count]]=1
                    pr=measure(y,p,t,alert);policyrows.append(dict(week=w,block=block,model=name,policy=policy,n=len(y),alerts=int((p>=t).sum()) if alert is None else int(alert.sum()),**pr))
                out=test[KEY+['imd_group','presentation_type','risk']].copy();out['probability']=p;out['week']=w;out['block']=block;out['model']=name;out['split']='2014_test';out['threshold']=threshold;predictions.append(out)
                joblib.dump({'pipeline':m,'calibrator':cal,'threshold':threshold},ROOT/f'outputs/models/w{w}_{block}_{name}.joblib')
                if w==4 and block=='D' and name in ['XGBoost','LightGBM'] and score>bestval:bestval=score;best=dict(model=name,validation_AUC=score,path=f'outputs/models/w4_D_{name}.joblib')
                print(f'ML week={w} block={block} model={name}: AUC={met["ROC_AUC"]:.4f}',flush=True)
        for name in names:
            for a,b in [('C','B'),('D','C'),('B','A')]:
                if (a,name) not in draw_cache or (b,name) not in draw_cache:continue
                for metric in ['ROC_AUC','average_precision','Brier']:
                    delta=draw_cache[(a,name)][metric]-draw_cache[(b,name)][metric];point=next(r[metric] for r in rows if r['week']==w and r['block']==a and r['model']==name)-next(r[metric] for r in rows if r['week']==w and r['block']==b and r['model']==name)
                    lo,hi=delta.quantile([.025,.975]);paired.append(dict(week=w,model=name,contrast=a+' minus '+b,metric=metric,difference=point,ci_low=lo,ci_high=hi))
    if not smoke:
        df=pd.read_parquet(ROOT/'data/processed/week_4.parquet')
        for module in sorted(df.code_module.unique()):
            test=df[df.code_module==module];train=df[(df.code_module!=module)&~df.id_student.isin(test.id_student)]
            f,v,c=split2013(train)
            for label,a in [('fit',f),('validation',v),('calibration',c),('test',test)]:
                s=a[KEY].copy();s['week']=4;s['evaluation']='LOMO_'+module;s['split']=label;splits.append(s)
            for name in ['Logistic',best['model']]:
                m=pipeline(f,'D',name,1 if name=='Logistic' else 2);m.fit(pd.concat([f,v]),pd.concat([f,v]).risk)
                p=m.predict_proba(test)[:,1];y=test.risk.to_numpy();b=bootstrap(y,p,test.id_student.to_numpy());r=dict(week=4,block='D',model=name,evaluation='LOMO',held_out_module=module,n=len(test),train_n=len(train),overlap_excluded=int(((df.code_module!=module)&df.id_student.isin(test.id_student)).sum()),prevalence=y.mean(),**measure(y,p))
                for metric in METRICS:r[metric+'_low'],r[metric+'_high']=b[metric].quantile([.025,.975])
                general.append(r);out=test[KEY+['imd_group','presentation_type','risk']].copy();out['probability']=p;out['week']=4;out['block']='D';out['model']=name;out['split']='LOMO_'+module;predictions.append(out)
                print('LOMO',module,name,flush=True)
    tag='_smoke' if smoke else ''
    table('T6_performance'+tag,pd.DataFrame(rows));table('T6_paired_differences'+tag,pd.DataFrame(paired));table('T7_generalisation'+tag,pd.DataFrame(general));table('T8_subgroups'+tag,pd.DataFrame(subrows));table('T8_alert_policies'+tag,pd.DataFrame(policyrows))
    pd.concat(predictions).to_parquet(ROOT/f'outputs/models/predictions{tag}.parquet',index=False);pd.concat(splits).to_csv(ROOT/f'reports/split_assignments{tag}.csv',index=False);pd.DataFrame(tuning).to_csv(ROOT/f'reports/tuning{tag}.csv',index=False)
    if best:(ROOT/'reports/selected_boosting.json').write_text(json.dumps(best,indent=2))
if __name__=='__main__':
    import sys
    run('--smoke' in sys.argv)
