# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

import pandas as pd,numpy as np
import statsmodels.api as sm,statsmodels.formula.api as smf
from association import BASE,fit
from common import *
def run():
    df=pd.read_parquet(ROOT/'data/processed/week_4.parquet')
    full=fit(df);reduced=smf.glm('success ~ C(imd_group)+active_proportion+'+BASE,data=df,family=sm.families.Binomial()).fit(maxiter=100,cov_type='cluster',cov_kwds={'groups':df.id_student.to_numpy()})
    idx=[t for t in full.params.index if ':' in t and 'imd_group' in t];test=named_wald(full,idx)
    table('T4_full_reduced_comparison',pd.DataFrame([dict(specification=label,n=len(df),parameters=len(m.params),log_likelihood=m.llf,AIC=m.aic,converged=m.converged,joint_interaction_Wald=float(test.statistic) if label=='with_interaction' else None,interaction_df=len(idx) if label=='with_interaction' else None,interaction_p=float(test.pvalue) if label=='with_interaction' else None) for label,m in [('without_interaction',reduced),('with_interaction',full)]]))
if __name__=='__main__':run()
