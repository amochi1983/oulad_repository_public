# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import checked_variance

def test_nonfinite_or_negative_variance_is_rejected():
    for v in [np.nan,np.inf,-1e-3]:
        with pytest.raises(ValueError):checked_variance(v)
    assert checked_variance(-1e-15)==0

def test_required_source_and_data_exist():
    for path in ['data/raw/studentInfo.csv','data/raw/studentRegistration.csv','data/raw/studentVle.csv','data/processed/week_4.parquet','reports/split_assignments.csv','reference_tables/outputs/tables/T4_models.csv']:
        assert (ROOT/path).is_file(),f'Missing required input: {path}'

def test_primary_model_numeric_regression():
    now=pd.read_csv(ROOT/'outputs/tables/T4_models.csv');ref=pd.read_csv(ROOT/'reference_tables/outputs/tables/T4_models.csv')
    a=now[now.analysis.eq('primary')].sort_values('term');b=ref[ref.analysis.eq('primary')].sort_values('term')
    assert len(a)==51 and a.converged.all()
    assert a.term.tolist()==b.term.tolist()
    np.testing.assert_allclose(a.estimate,b.estimate,rtol=1e-7,atol=1e-9)
    np.testing.assert_allclose(a.se,b.se,rtol=1e-7,atol=1e-9)

def test_reported_gaps_and_module_results_reproduce():
    for folder,name,key,metrics in [
        ('outputs/first_project/tables','S7_adjusted_success_gaps.csv',['contrast','engagement'],['gap','ci_low','ci_high']),
        ('outputs/first_project/tables','S3_adjusted_engagement_associations.csv',['analysis','imd_group'],['OR_per_7_active_days','OR_ci_low','OR_ci_high']),
        ('outputs/manuscript_checks','module_OR.csv',['module','group'],['OR','low','high']),
        ('outputs/manuscript_checks','unknown_withdrawal_gaps.csv',['analysis','active_days'],['gap','low','high'])]:
        a=pd.read_csv(ROOT/folder/name);b=pd.read_csv(ROOT/'reference_tables'/folder/name)
        # Fail explicitly when expected artifacts disappear; never silently pass.
        assert len(a)==len(b)>0
        assert set(key).issubset(a.columns)
        a=a.sort_values(key).reset_index(drop=True);b=b.sort_values(key).reset_index(drop=True)
        cols=[c for c in metrics if c in a and c in b]
        assert cols
        np.testing.assert_allclose(a[cols],b[cols],rtol=1e-7,atol=1e-9)

def test_core_retraining_is_complete_and_exact():
    p=ROOT/'reports/core_retraining/core_retraining_verification.json'
    assert p.is_file(),'Run reproduce.py --retrain-core'
    d=json.loads(p.read_text(encoding='utf-8-sig'))
    # Registry schema is retained in the source file; this checks actual exact-reproduction verdict.
    assert d.get('all_match_within_1e_10') is True


def test_journal_figures_have_rgb_dpi_size_and_source_data():
    from PIL import Image
    names=['adjusted_engagement_success_OR','adjusted_success_probabilities','adjusted_success_gaps','module_engagement_success_OR']
    for name in names:
        p=ROOT/f'outputs/first_project/figures/{name}.tif'
        assert p.is_file() and (p.with_suffix('.csv')).is_file()
        with Image.open(p) as im:
            assert im.mode=='RGB'
            assert 789<=im.width<=2250 and im.height<=2625
            assert all(abs(float(v)-300)<1 for v in im.info['dpi'])
        assert p.stat().st_size<10_000_000
