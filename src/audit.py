# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

import duckdb,pandas as pd
from common import *
def run():
    con=duckdb.connect();raw=(ROOT/'data/raw').as_posix();rows=[]
    queries={
    'vle_duplicate_student_resource_day':f"SELECT COUNT(*) n FROM (SELECT id_student,code_module,code_presentation,id_site,date,COUNT(*) k FROM read_csv_auto('{raw}/studentVle.csv') GROUP BY ALL HAVING COUNT(*)>1)",
    'vle_unmatched_enrolments':f"SELECT COUNT(*) n FROM read_csv_auto('{raw}/studentVle.csv') l LEFT JOIN read_csv_auto('{raw}/studentInfo.csv') i USING(id_student,code_module,code_presentation) WHERE i.id_student IS NULL",
    'assessment_unmatched_metadata':f"SELECT COUNT(*) n FROM read_csv_auto('{raw}/studentAssessment.csv') s LEFT JOIN read_csv_auto('{raw}/assessments.csv') a USING(id_assessment) WHERE a.id_assessment IS NULL",
    'assessment_unmatched_enrolments':f"SELECT COUNT(*) n FROM read_csv_auto('{raw}/studentAssessment.csv') s JOIN read_csv_auto('{raw}/assessments.csv') a USING(id_assessment) LEFT JOIN read_csv_auto('{raw}/studentInfo.csv') i USING(id_student,code_module,code_presentation) WHERE i.id_student IS NULL",
    'vle_missing_required_fields':f"SELECT COUNT(*) n FROM read_csv_auto('{raw}/studentVle.csv') WHERE id_student IS NULL OR code_module IS NULL OR code_presentation IS NULL OR id_site IS NULL OR date IS NULL OR sum_click IS NULL",
    'unregistration_before_registration':f"SELECT COUNT(*) n FROM read_csv_auto('{raw}/studentRegistration.csv') WHERE date_unregistration<date_registration",
    'vle_after_course_length':f"SELECT COUNT(*) n FROM read_csv_auto('{raw}/studentVle.csv') l JOIN read_csv_auto('{raw}/courses.csv') c USING(code_module,code_presentation) WHERE l.date>=c.module_presentation_length"}
    for name,q in queries.items():rows.append(dict(check=name,n=con.sql(q).fetchone()[0]))
    table('T1_relational_date_checks',pd.DataFrame(rows))
    schema=con.sql(f"DESCRIBE SELECT * FROM read_csv_auto('{raw}/studentVle.csv')").df()
    table('T1_vle_schema',schema)
    con.close()
if __name__=='__main__':run()
