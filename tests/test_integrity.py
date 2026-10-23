# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import *
def test_landmark_boundaries():
    d=pd.DataFrame({'date_registration':[-1,7,6,-1,np.nan],'date_unregistration':[np.nan,np.nan,7,6,np.nan]})
    assert eligible(d,1).tolist()==[True,False,True,False,False]
def test_imd_labels():
    assert [imd_group(v) for v in ['0-10%','20-30%','30-40%','60-70%','90-100%',np.nan]]==['0–30%','0–30%','30–70%','30–70%','70–100%','Missing']
def test_real_cohorts_and_zero_activity():
    p=ROOT/'data/processed/base.parquet'
    assert p.exists(), f'Required integration input absent: {p}; run reproduce.py --rebuild-features'
    base=pd.read_parquet(p)
    assert not base.duplicated(KEY).any()
    for w in CFG['weeks']:
        df=pd.read_parquet(ROOT/f'data/processed/week_{w}.parquet')
        assert len(df)==eligible(base,w).sum()
        assert not df.duplicated(KEY).any()
        assert df.active_days.between(0,7*w).all()
        assert (df.loc[df.total_clicks==0,'active_days']==0).all()
        assert df.total_clicks.eq(0).any()
        assert not any(c in df for c in ['engagement_slope','recent_change']) if w==1 else True
def test_temporal_split():
    p=ROOT/'reports/split_assignments.csv'
    assert p.exists(), f'Required integration input absent: {p}; run reproduce.py --rebuild-features'
    d=pd.read_csv(p)
    for _,a in d.groupby(['week','evaluation']):
        sets={s:set(g.id_student) for s,g in a.groupby('split')}
        names=list(sets)
        for i,s in enumerate(names):
            for t in names[i+1:]:assert sets[s].isdisjoint(sets[t])
def test_table_denominators():
    p=ROOT/'outputs/tables/T1_cohorts.csv'
    assert p.exists(), f'Required integration input absent: {p}; run reproduce.py --rebuild-features'
    d=pd.read_csv(p)
    for w in CFG['weeks']:assert d.loc[d.week==w,'enrolments'].sum()==len(pd.read_parquet(ROOT/f'data/processed/week_{w}.parquet'))
def test_temporal_feature_totals_against_raw():
    p=ROOT/'data/processed/week_1.parquet'
    assert p.exists(), f'Required integration input absent: {p}; run reproduce.py --rebuild-features'
    import duckdb
    df=pd.read_parquet(p);sample=df.sample(8,random_state=42)
    con=duckdb.connect()
    for _,r in sample.iterrows():
        q=f"SELECT coalesce(sum(sum_click),0),count(distinct CASE WHEN sum_click>0 THEN date END) FROM read_csv_auto('{(ROOT/'data/raw/studentVle.csv').as_posix()}') WHERE id_student={r.id_student} AND code_module='{r.code_module}' AND code_presentation='{r.code_presentation}' AND date>=0 AND date<7"
        clicks,days=con.sql(q).fetchone();assert clicks==r.total_clicks;assert days==r.active_days
def test_fast_cluster_bootstrap_matches_metrics():
    from ml import bootstrap,measure
    old=CFG['bootstrap'];CFG['bootstrap']=1
    y=np.array([0,1,1,0,1,0]);p=np.array([.2,.8,.6,.6,.9,.1]);ids=np.array([1,1,2,3,4,4]);rng=np.random.default_rng(CFG['seed']);groups=pd.factorize(ids)[0]
    draws=rng.integers(0,4,4);ix=np.concatenate([np.flatnonzero(groups==i) for i in draws]);expected=measure(y[ix],p[ix]);actual=bootstrap(y,p,ids).iloc[0]
    for m in expected:assert np.isclose(expected[m],actual[m],equal_nan=True),m
    CFG['bootstrap']=old
