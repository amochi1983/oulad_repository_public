# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

"""Regression checks for clustered resampling and named constraints."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from common import student_descriptive_bootstrap,named_wald
from features import refresh_click_features

class ScientificCorrections(unittest.TestCase):
    def test_cluster_draws_keep_repeat_enrolments_together(self):
        d=pd.DataFrame({'id_student':[1,1,2,3], 'success':[1,0,0,1], 'active_proportion':[.5,.5,0,1]})
        masks={'all':np.ones(4,dtype=bool)}
        result=student_descriptive_bootstrap(d,masks,replicates=30)
        rng=np.random.default_rng(20261005)
        expected=[]
        for _ in range(30):
            sampled=rng.integers(0,3,3)
            records=pd.concat([d[d.id_student==i+1] for i in sampled])
            expected.append(records.success.mean())
        np.testing.assert_allclose(result.all_success,expected)

    def test_named_wald_survives_formula_reordering(self):
        rng=np.random.default_rng(4)
        d=pd.DataFrame({'a':rng.normal(size=100),'b':rng.normal(size=100),'y':rng.normal(size=100)})
        first=smf.ols('y ~ a + b',d).fit()
        second=smf.ols('y ~ b + a',d).fit()
        np.testing.assert_allclose(named_wald(first,['a']).statistic,named_wald(second,['a']).statistic)
        with self.assertRaises(ValueError):named_wald(first,['absent'])

    def test_clean_click_counts_drive_all_volume_features(self):
        d=pd.DataFrame({'total_clicks':[10.,0.],'active_days':[2.,0.],
                        'clicks_week_1':[8.,0.],'clicks_week_2':[2.,0.]})
        refresh_click_features(d,2)
        np.testing.assert_allclose(d.log_clicks,np.log1p([10,0]))
        np.testing.assert_allclose(d.clicks_per_active_day,[5,0])
        np.testing.assert_allclose(d.recent_change,[-6,0])

    def test_ale_uses_supported_profiles_and_coherent_derivatives(self):
        from explain import coherent_engagement
        d=pd.DataFrame({'code_module':['AAA','AAA'],'imd_group':['group','group'],
            'id_student':[1,2],'active_days':[7.,14.],'active_proportion':[.25,.5],
            'active_weeks':[1,2],'resources_accessed':[2,3],'activity_diversity':[1,2],
            'days_since_activity':[21,14],'total_clicks':[14.,42.],
            'days_week_1':[7.,7.],'days_week_2':[0.,7.],'days_week_3':[0.,0.],'days_week_4':[0.,0.],
            'clicks_week_1':[14.,21.],'clicks_week_2':[0.,21.],'clicks_week_3':[0.,0.],'clicks_week_4':[0.,0.]})
        refresh_click_features(d,4)
        updated=coherent_engagement(d.iloc[[0]],.5,d)
        self.assertEqual(updated.active_days.iloc[0],14)
        self.assertEqual(updated.total_clicks.iloc[0],42)
        self.assertEqual(updated.clicks_per_active_day.iloc[0],3)
        np.testing.assert_allclose(updated.log_clicks,np.log1p([42]))
        self.assertEqual(updated.id_student.iloc[0],1)

if __name__=='__main__':unittest.main()
