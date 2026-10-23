# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

import duckdb
import pandas as pd
import numpy as np
from common import *
def refresh_click_features(df, weeks):
    """Derive all volume features from the currently selected click counts."""
    df['log_clicks'] = np.log1p(df.total_clicks)
    df['clicks_per_active_day'] = (df.total_clicks / df.active_days.replace(0, np.nan)).fillna(0)
    if weeks >= 2:
        ar = df[[f'clicks_week_{j+1}' for j in range(weeks)]].to_numpy()
        x = np.arange(weeks) - (weeks-1)/2
        df['engagement_slope'] = ar @ x / (x @ x)
        df['engagement_sd'] = ar.std(axis=1, ddof=1)
        df['recent_change'] = ar[:, -1] - ar[:, -2]
    return df

def build():
    raw=ROOT/'data/raw';con=duckdb.connect(str(ROOT/'data/processed/aggregate.duckdb'))
    con.execute("SET threads=2")
    info=pd.read_csv(raw/'studentInfo.csv');reg=pd.read_csv(raw/'studentRegistration.csv')
    base=info.merge(reg,on=KEY,how='left',validate='one_to_one',indicator=True)
    assert (base['_merge']=='both').all();base=base.drop(columns='_merge')
    base['imd_group']=base.imd_band.map(imd_group);base['success']=base.final_result.isin(['Pass','Distinction']).astype(int)
    base['risk']=1-base.success;base['withdrawal']=base.final_result.eq('Withdrawn').astype(int)
    base['context']=base.code_module+'_'+base.code_presentation;base['presentation_type']=base.code_presentation.str[-1]
    base['year']=base.code_presentation.str[:4].astype(int)
    base['inconsistent_withdrawal']=base.final_result.eq('Withdrawn')!=base.date_unregistration.notna()
    base['registration_lag']=base.date_registration.clip(lower=0)
    base.to_parquet(ROOT/'data/processed/base.parquet',index=False)
    con.execute(f"CREATE OR REPLACE VIEW logs AS SELECT * FROM read_csv_auto('{raw.as_posix()}/studentVle.csv')")
    con.execute(f"CREATE OR REPLACE VIEW resources AS SELECT * FROM read_csv_auto('{raw.as_posix()}/vle.csv')")
    daily=ROOT/'data/processed/daily.parquet'
    # Rebuild from the current raw source rather than reusing a stale aggregate.
    con.execute(f"COPY (SELECT id_student,code_module,code_presentation,date,SUM(sum_click) clicks FROM logs GROUP BY ALL) TO '{daily.as_posix()}' (FORMAT PARQUET)")
    audit=[]
    keys={'studentInfo':KEY,'studentRegistration':KEY,'courses':['code_module','code_presentation'],'vle':['id_site','code_module','code_presentation'],'assessments':['id_assessment'],'studentAssessment':['id_assessment','id_student']}
    for name,k in keys.items():
        df=pd.read_csv(raw/(name+'.csv'))
        for col in df:
            audit.append(dict(file=name,column=col,dtype=str(df[col].dtype),rows=len(df),missing=int(df[col].isna().sum()),duplicate_keys=int(df.duplicated(k).sum())))
        assert not df.duplicated(k).any(),f'Duplicate key in {name}'
    table('T1_schema_audit',pd.DataFrame(audit))
    checks=con.sql("SELECT COUNT(*) AS n_rows, SUM(CASE WHEN sum_click<0 THEN 1 ELSE 0 END) negative_clicks, MIN(date) min_day,MAX(date) max_day FROM logs").df()
    table('T1_vle_audit',checks)
    unmatched=con.sql('SELECT COUNT(*) unmatched FROM logs l LEFT JOIN resources r USING(id_site,code_module,code_presentation) WHERE r.id_site IS NULL').df()
    assert unmatched.unmatched.iloc[0]==0
    table('T1_join_audit',unmatched)
    assessments=pd.read_csv(raw/'studentAssessment.csv').merge(pd.read_csv(raw/'assessments.csv'),on='id_assessment',validate='many_to_one')
    assessments=assessments[(assessments.assessment_type!='Exam') & (assessments.is_banked==0)]
    flows=[]
    for w in CFG['weeks']:
        cut=7*w;df=base.loc[eligible(base,w)].copy()
        agg=con.sql(f"""SELECT l.id_student,l.code_module,l.code_presentation,
        SUM(CASE WHEN date>=0 THEN sum_click ELSE 0 END) total_clicks,
        COUNT(DISTINCT CASE WHEN date>=0 AND sum_click>0 THEN date END) active_days,
        COUNT(DISTINCT CASE WHEN date>=0 AND sum_click>0 THEN floor(date/7) END) active_weeks,
        COUNT(DISTINCT CASE WHEN date>=0 AND sum_click>0 THEN id_site END) resources_accessed,
        COUNT(DISTINCT CASE WHEN date>=0 AND sum_click>0 THEN activity_type END) activity_diversity,
        MAX(CASE WHEN date>=0 AND sum_click>0 THEN date END) last_day,
        SUM(CASE WHEN date<0 THEN sum_click ELSE 0 END) precourse_clicks,
        COUNT(DISTINCT CASE WHEN date<0 AND sum_click>0 THEN date END) precourse_active_days
        FROM logs l LEFT JOIN resources r USING(id_site,code_module,code_presentation) WHERE date<{cut} GROUP BY l.id_student,l.code_module,l.code_presentation""").df()
        df=df.merge(agg,on=KEY,how='left',validate='one_to_one')
        cols=['total_clicks','active_days','active_weeks','resources_accessed','activity_diversity','precourse_clicks','precourse_active_days']
        df[cols]=df[cols].fillna(0)
        df['days_since_activity']=(cut-1-df.last_day).fillna(cut);df=df.drop(columns='last_day')
        df['log_clicks']=np.log1p(df.total_clicks);df['active_proportion']=df.active_days/cut
        df['clicks_per_active_day']=df.total_clicks/df.active_days.replace(0,np.nan);df['clicks_per_active_day']=df.clicks_per_active_day.fillna(0)
        for j in range(w):
            a=con.sql(f"SELECT id_student,code_module,code_presentation,SUM(clicks) weekly_clicks,COUNT(DISTINCT CASE WHEN clicks>0 THEN date END) weekly_days FROM read_parquet('{daily.as_posix()}') WHERE date>={7*j} AND date<{7*(j+1)} GROUP BY ALL").df()
            a=a.rename(columns={'weekly_clicks':f'clicks_week_{j+1}','weekly_days':f'days_week_{j+1}'})
            df=df.merge(a,on=KEY,how='left',validate='one_to_one');df[[f'clicks_week_{j+1}',f'days_week_{j+1}']]=df[[f'clicks_week_{j+1}',f'days_week_{j+1}']].fillna(0)
        if w>=2:
            ar=df[[f'clicks_week_{j+1}' for j in range(w)]].to_numpy();x=np.arange(w);x=x-x.mean()
            df['engagement_slope']=ar@x/(x@x);df['engagement_sd']=ar.std(axis=1,ddof=1);df['recent_change']=ar[:,-1]-ar[:,-2]
        refresh_click_features(df,w)
        a=assessments[assessments.date_submitted<cut].groupby(KEY).agg(observed_submissions=('id_assessment','nunique'),score_sensitivity=('score','mean')).reset_index()
        df=df.merge(a,on=KEY,how='left',validate='one_to_one');df['observed_submissions']=df.observed_submissions.fillna(0)
        df.to_parquet(ROOT/f'data/processed/week_{w}.parquet',index=False)
        for (g,o),a in df.groupby(['imd_group','final_result']):flows.append(dict(week=w,imd_group=g,outcome=o,enrolments=len(a),unique_students=a.id_student.nunique(),zero_clicks=int(a.total_clicks.eq(0).sum())))
        print(f'Features week {w}: {len(df)} eligible enrolments',flush=True)
    table('T1_cohorts',pd.DataFrame(flows))
    table('T1_date_consistency',base.groupby(['final_result','inconsistent_withdrawal'],dropna=False).agg(enrolments=('id_student','size'),unique_students=('id_student','nunique')).reset_index())
    con.close()
if __name__=='__main__':build()
