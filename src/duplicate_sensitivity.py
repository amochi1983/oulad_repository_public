# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

import duckdb,pandas as pd,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import *
from ml import pipeline,split2013,calibrated,measure,bootstrap
from association import fit
from features import refresh_click_features
import hashlib,json
def run():
    con=duckdb.connect();raw=(ROOT/'data/raw/studentVle.csv').as_posix();df=pd.read_parquet(ROOT/'data/processed/week_4.parquet')
    select=['id_student','code_module','code_presentation','SUM(CASE WHEN date>=0 THEN sum_click ELSE 0 END) dedup_clicks','SUM(CASE WHEN date<0 THEN sum_click ELSE 0 END) dedup_precourse']+[f'SUM(CASE WHEN date>={7*j} AND date<{7*(j+1)} THEN sum_click ELSE 0 END) dedup_week_{j+1}' for j in range(4)]
    a=con.sql(f"SELECT {','.join(select)} FROM (SELECT DISTINCT * FROM read_csv_auto('{raw}') WHERE date<28) GROUP BY id_student,code_module,code_presentation").df();df=df.merge(a,on=KEY,how='left',validate='one_to_one');columns=[c for c in df if c.startswith('dedup_')];df[columns]=df[columns].fillna(0)
    original=df[KEY+['total_clicks','log_clicks']].copy().rename(columns={'total_clicks':'raw_clicks','log_clicks':'raw_log_clicks'})
    df['total_clicks']=df.dedup_clicks;df['log_clicks']=np.log1p(df.total_clicks);df['precourse_clicks']=df.dedup_precourse;df['clicks_per_active_day']=(df.total_clicks/df.active_days.replace(0,np.nan)).fillna(0)
    ar=df[[f'dedup_week_{j+1}' for j in range(4)]].to_numpy();x=np.arange(4)-1.5
    for j in range(4):df[f'clicks_week_{j+1}']=ar[:,j]
    df['engagement_slope']=ar@x/(x@x);df['engagement_sd']=ar.std(axis=1,ddof=1);df['recent_change']=ar[:,-1]-ar[:,-2]
    refresh_click_features(df,4)
    assert np.allclose(df.log_clicks,np.log1p(df.dedup_clicks))
    audit=original.merge(df[KEY+['total_clicks','log_clicks']],on=KEY,validate='one_to_one')
    audit['removed_clicks']=audit.raw_clicks-audit.total_clicks
    assert audit.removed_clicks.ge(0).all()
    audit.to_csv(ROOT/'reports/duplicate_feature_audit.csv',index=False)
    table('T8_duplicate_feature_audit',audit.groupby('code_module').agg(enrolments=('id_student','size'),affected_enrolments=('removed_clicks',lambda x:int(x.gt(0).sum())),raw_clicks=('raw_clicks','sum'),deduplicated_clicks=('total_clicks','sum'),removed_clicks=('removed_clicks','sum')).reset_index())
    with (ROOT/'data/raw/studentVle.csv').open('rb') as source:
        raw_sha256=hashlib.file_digest(source,'sha256').hexdigest()
    (ROOT/'reports/duplicate_sensitivity_manifest.json').write_text(json.dumps(dict(source='studentVle.csv',source_sha256=raw_sha256,deduplication='SELECT DISTINCT all raw columns before aggregation',affected_enrolments=int(audit.removed_clicks.gt(0).sum()),removed_clicks=float(audit.removed_clicks.sum()),derived_click_features='recomputed exclusively from deduplicated counts'),indent=2))
    m=fit(df,'log_clicks');ci=m.conf_int();table('T8_deduplicated_click_association',pd.DataFrame([dict(term=t,estimate=m.params[t],ci_low=ci.loc[t,0],ci_high=ci.loc[t,1],p=m.pvalues[t],n=len(df)) for t in m.params.index]))
    test=df[df.year==2014];train=df[(df.year==2013)&~df.id_student.isin(test.id_student)];f,v,c=split2013(train);best=-np.inf;chosen=None
    for param in [.3,1]:
        pipe=pipeline(f,'D','Logistic',param);pipe.fit(f,f.risk);auc=roc_auc_score(v.risk,pipe.predict_proba(v)[:,1])
        if auc>best:best=auc;chosen=param
    pipe=pipeline(f,'D','Logistic',chosen);fitall=pd.concat([f,v]);pipe.fit(fitall,fitall.risk);pc=pipe.predict_proba(c)[:,1];z=np.log(np.clip(pc,1e-6,1-1e-6)/np.clip(1-pc,1e-6,1))[:,None];cal=LogisticRegression(C=1e3,max_iter=1000).fit(z,c.risk);p=calibrated(pipe.predict_proba(test)[:,1],cal);b=bootstrap(test.risk.to_numpy(),p,test.id_student.to_numpy());r=dict(analysis='exact_row_deduplication_sensitivity',model='Logistic',block='D',week=4,n=len(test),validation_AUC=best,**measure(test.risk.to_numpy(),p))
    for metric in b:r[metric+'_low'],r[metric+'_high']=b[metric].quantile([.025,.975])
    table('T8_deduplicated_click_prediction',pd.DataFrame([r]));out=test[KEY+['risk']].copy();out['probability']=p;out.to_csv(ROOT/'outputs/models/deduplicated_click_predictions.csv',index=False)
if __name__=='__main__':run()
