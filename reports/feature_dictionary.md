# Feature dictionary

For enrolment i at week w, cutoff c=7w. In-course events have 0≤date<c. Positive click records define activity. All aggregations use student/module/presentation keys; resource metadata joins site/module/presentation keys. Missing activity after a left join is zero, not an excluded enrolment.

| Feature | Definition |
|---|---|
| total_clicks | Sum of recorded clicks for days 0,…,c−1 |
| log_clicks | ln(1+total_clicks) |
| active_days | Number of distinct days with >0 clicks |
| active_weeks | Number of distinct floor(date/7) with >0 clicks |
| active_proportion | active_days/c; primary exposure; full calendar-window denominator |
| clicks_per_active_day | total_clicks/active_days; 0 when active_days=0 |
| resources_accessed | Distinct id_site with positive clicks in course window |
| activity_diversity | Distinct activity_type with positive clicks |
| days_since_activity | c−1−max(active date); c if no in-course activity |
| clicks_week_j | Click sum in [7(j−1),7j) |
| days_week_j | Positive-click active-day count in same weekly interval |
| engagement_slope | OLS slope of weekly clicks against week index: Σ(j−j̄)Cj / Σ(j−j̄)²; w≥2 only |
| engagement_sd | Sample standard deviation of observed weekly clicks; w≥2 |
| recent_change | Cw−C(w−1); w≥2 |
| precourse_clicks | Click sum on dates<0 and date<c, separate from in-course totals |
| precourse_active_days | Distinct pre-course positive-click dates |
| observed_submissions | Distinct non-Exam, non-banked assessments submitted before c, after metadata join |
| score_sensitivity | Mean score among those submissions; release dates unknown; sensitivity only |

Baseline predictors: gender, age_band, highest_education, disability, region, num_of_prev_attempts, studied_credits, module, presentation type B/J; blocks B–D add grouped IMD. Association analysis instead includes module/presentation fixed effects. final_result defines outcomes only. date_unregistration defines eligibility and consistency only. year defines temporal split only. No future presentation ID, final outcome, unregistration date, or inconsistency flag enters predictive preprocessing.

Labels found: `0-10%`, `10-20` (missing percent sign in source), `20-30%`, …, `90-100%`. Parse the lower numeric bound; [0,30), [30,70), [70,100) yield the prespecified groups. Original bands remain unchanged in raw and descriptive files. Missing IMD is separate. Lower bands indicate greater area deprivation. National definitions and scales differ; region adjustment does not establish UK-wide comparability or individual income.
