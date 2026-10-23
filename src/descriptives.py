# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

import duckdb,pandas as pd,numpy as np
from statsmodels.stats.proportion import proportion_confint
from common import *
def run():
    df=pd.read_parquet(ROOT/'data/processed/week_4.parquet');rows=[]
    for g,a in df.groupby('imd_band',dropna=False):
        r=dict(imd_band='Missing' if pd.isna(g) else g,n=len(a),students=a.id_student.nunique(),successes=int(a.success.sum()),success_rate=a.success.mean(),median_clicks=a.total_clicks.median(),median_active_proportion=a.active_proportion.median(),clicks_p25=a.total_clicks.quantile(.25),clicks_p75=a.total_clicks.quantile(.75),active_p25=a.active_proportion.quantile(.25),active_p75=a.active_proportion.quantile(.75));r['ci_low'],r['ci_high']=proportion_confint(r['successes'],r['n'],method='wilson');rows.append(r)
    table('T3_original_IMD_bands',pd.DataFrame(rows))
    table('T3_four_outcomes_by_IMD',df.groupby(['imd_group','final_result']).agg(enrolments=('id_student','size'),unique_students=('id_student','nunique')).reset_index())
    con=duckdb.connect();raw=(ROOT/'data/raw/studentVle.csv').as_posix()
    d=con.sql(f"SELECT COUNT(*) n_rows,COUNT(DISTINCT (id_student,code_module,code_presentation,id_site,date,sum_click)) distinct_full_rows FROM read_csv_auto('{raw}')").df();d['exact_duplicate_excess']=d.n_rows-d.distinct_full_rows;table('T1_vle_exact_duplicates',d)
    # Primary active-day measure invariant to exact-row multiplicity.
    (ROOT/'reports/duplicate_handling.md').write_text('studentVle has repeated student/module/presentation/resource/day combinations; this combination is not assumed to be a unique event key. Raw rows are retained and click counts summed as documented; active days/resources are distinct counts. The exact-row multiplicity audit is exported. Without event identifiers it is not possible to distinguish identical events from duplication artifacts. The primary active-day exposure is invariant to exact-row deduplication; click features and log-click sensitivity may be affected. Rows on/after course length are audited but cannot enter any early cutoff (maximum 84 days).\n',encoding='utf-8')
if __name__=='__main__':run()
