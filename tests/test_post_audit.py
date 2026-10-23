# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi
"""Regression checks anchored to the separate calculations in the Claude (Anthropic)-assisted internal audit; not independent human peer review."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[1]

def test_post_audit_contrasts_match_independent_recalculation():
    d=pd.read_csv(ROOT/'outputs/post_audit/adjustment_contrasts.csv');a=d[d.active_days.eq(21)].set_index('analysis')
    expected={'full':[.07587648340146991,.05342383435120043,.09832913245173827],'context_only':[.10078561837875455,.07857993759784054,.12299129915966855],'no_education_no_prev_attempts':[.0884599621243014,.06597288406695678,.11094704018164603],'no_region':[.08428343507512415,.06212470619822574,.10644216395202255],'full_known_IMD_reference':[.07551419342921406,.05316512608303496,.09786326077539316]}
    for key,values in expected.items():np.testing.assert_allclose(a.loc[key,['gap','ci_low','ci_high']].astype(float),values,atol=1e-8,rtol=0)
    assert (a.loc[list(expected)[:4],'fit_n']==27528).all()
    assert a.loc['english_regions_only','fit_n']==21352
    np.testing.assert_allclose(a.loc['english_regions_only','gap'],.07758891883516117,atol=1e-8)

def test_reported_calibration_has_correct_score_equations():
    d=pd.read_parquet(ROOT/'outputs/models/predictions.parquet');c=pd.read_csv(ROOT/'outputs/post_audit/test_calibration.csv')
    for row in c.itertuples():
        a=d[d.week.eq(4)&d.model.eq('Logistic')&d.block.eq(row.block)&d.split.eq('2014_test')];z=logit(a.probability.to_numpy());y=a.risk.to_numpy()
        assert len(a)==row.n==15722
        assert abs(np.mean(y-expit(z+row.calibration_in_large)))<1e-8
        errors=y-expit(row.joint_intercept+row.slope*z)
        assert abs(errors.mean())<1e-8 and abs(np.mean(errors*z))<1e-8
        assert np.isclose(np.mean((a.probability-a.risk)**2),row.Brier,atol=1e-12)
