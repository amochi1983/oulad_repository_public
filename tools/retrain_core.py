# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

"""Reproduce only manuscript week-four B/C/D penalised logistic training.

All outputs are separate; no archived model/table is overwritten. Inputs must exist.
Usage: python retrain_core.py --repository-root C:/Oulad --output-dir PATH
For a clean code copy with external archived references, add --reference-root PATH.
"""
from pathlib import Path
import argparse, sys, json, time, hashlib, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root',type=Path,required=True)
    parser.add_argument('--reference-root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); root=args.repository_root.resolve(); ref=(args.reference_root or root).resolve(); out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'src'))
    import ml
    started=time.time()
    paths={'cohort':root/'data/processed/week_4.parquet','splits':ref/'reports/split_assignments.csv','predictions':ref/'outputs/models/predictions.parquet','performance':ref/'outputs/tables/T6_performance.csv','source_ml':root/'src/ml.py','config':root/'configs/analysis.json'}
    for name,path in paths.items():
        if not path.is_file(): raise FileNotFoundError(f'Required {name}: {path}')
    df=pd.read_parquet(paths['cohort']); assignment=pd.read_csv(paths['splits']); assignment=assignment[(assignment.week==4)&assignment.evaluation.eq('temporal')]
    key=['id_student','code_module','code_presentation']; assert not assignment.duplicated(key).any()
    parts={label:df.merge(assignment.loc[assignment.split.eq(label),key],on=key,how='inner',validate='one_to_one') for label in ['fit','validation','calibration','test']}
    ids={name:set(part.id_student) for name,part in parts.items()}
    assert all(ids[a].isdisjoint(ids[b]) for a in ids for b in ids if a!=b)
    assert all(part.year.eq(2013).all() for name,part in parts.items() if name!='test') and parts['test'].year.eq(2014).all()
    expected_test=df[df.year.eq(2014)]; expected_dev=df[df.year.eq(2013)&~df.id_student.isin(expected_test.id_student)]
    assert set(map(tuple,parts['test'][key].to_numpy()))==set(map(tuple,expected_test[key].to_numpy()))
    assert set(map(tuple,pd.concat([parts[x] for x in ['fit','validation','calibration']])[key].to_numpy()))==set(map(tuple,expected_dev[key].to_numpy()))
    # Validate that the recorded partition/order matches the exact source splitting rule.
    current_parts=ml.split2013(expected_dev)
    for name,current in zip(['fit','validation','calibration'],current_parts):
        assert list(map(tuple,parts[name][key].to_numpy()))==list(map(tuple,current[key].to_numpy())),f'Saved/current {name} split differs'
    f,v,c,test=[parts[name] for name in ['fit','validation','calibration','test']]
    archived=pd.read_parquet(paths['predictions']);performance=pd.read_csv(paths['performance'])
    results=[];tuning=[];scores=[];boots={}
    for block in ['B','C','D']:
        chosen=None;best=-np.inf
        for param in [.3,1]:
            fitted=ml.pipeline(f,block,'Logistic',param)
            fitted.fit(f,f.risk)
            score=roc_auc_score(v.risk,fitted.predict_proba(v)[:,1])
            tuning.append({'block':block,'C':param,'validation_auc':score})
            if score>best: chosen=param;best=score
        # Source procedure refits preprocessing and estimator on fitting+validation.
        fitall=pd.concat([f,v]); fitted=ml.pipeline(fitall,block,'Logistic',chosen);fitted.fit(fitall,fitall.risk)
        pc=fitted.predict_proba(c)[:,1]
        z=np.log(np.clip(pc,1e-6,1-1e-6)/np.clip(1-pc,1e-6,1))[:,None]
        calibrator=LogisticRegression(C=1e3,max_iter=1000).fit(z,c.risk)
        p=ml.calibrated(fitted.predict_proba(test)[:,1],calibrator)
        saved=archived[(archived.week==4)&archived.block.eq(block)&archived.model.eq('Logistic')&archived.split.eq('2014_test')]
        aligned=test[key].merge(saved[key+['probability']],on=key,validate='one_to_one'); archived_p=aligned.probability.to_numpy()
        assert len(saved)==len(test) and np.isfinite(p).all()
        historical=performance[(performance.week==4)&performance.block.eq(block)&performance.model.eq('Logistic')].iloc[0]
        row={'block':block,'n':len(test),'students':test.id_student.nunique(),'chosen_C':chosen,'saved_chosen_C':float(historical.chosen_param),'validation_auc':best,'saved_validation_auc':float(historical.validation_AUC),'auc':roc_auc_score(test.risk,p),'saved_auc':roc_auc_score(test.risk,archived_p),'brier':brier_score_loss(test.risk,p),'saved_brier':brier_score_loss(test.risk,archived_p),'max_probability_difference':float(np.max(abs(p-archived_p))),'mean_probability_difference':float(np.mean(abs(p-archived_p)))}
        boot=ml.bootstrap(test.risk.to_numpy(),p,test.id_student.to_numpy());boots[block]=boot
        row['auc_ci_lower'],row['auc_ci_upper']=[float(x) for x in boot.ROC_AUC.quantile([.025,.975])]
        row['saved_auc_ci_lower']=float(historical.ROC_AUC_low);row['saved_auc_ci_upper']=float(historical.ROC_AUC_high)
        row['bootstrap_interval_match_within_1e_10']=bool(np.allclose([row['auc_ci_lower'],row['auc_ci_upper']],[historical.ROC_AUC_low,historical.ROC_AUC_high],atol=1e-10,rtol=0))
        row['match_within_1e_10']=bool(np.allclose(p,archived_p,atol=1e-10,rtol=0) and chosen==float(historical.chosen_param))
        results.append(row);q=test[key+['risk']].copy();q['block']=block;q['probability_retrained']=p;q['probability_archived']=archived_p;scores.append(q)
        print(json.dumps(row,default=int),flush=True)
    paired=[]
    saved_paired=pd.read_csv(ref/'outputs/tables/T6_paired_differences.csv')
    for high,low in [('C','B'),('D','C')]:
        delta=boots[high].ROC_AUC-boots[low].ROC_AUC;ci=delta.quantile([.025,.975]).to_numpy();saved=saved_paired[(saved_paired.week==4)&saved_paired.model.eq('Logistic')&saved_paired.contrast.eq(high+' minus '+low)&saved_paired.metric.eq('ROC_AUC')].iloc[0]
        paired.append({'contrast':high+' minus '+low,'lower':float(ci[0]),'upper':float(ci[1]),'saved_lower':float(saved.ci_low),'saved_upper':float(saved.ci_high),'match_within_1e_10':bool(np.allclose(ci,[saved.ci_low,saved.ci_high],atol=1e-10,rtol=0))})
    pd.DataFrame(tuning).to_csv(out/'core_retraining_tuning.csv',index=False)
    pd.concat(scores).to_csv(out/'core_retraining_scores.csv',index=False)
    manifest={'scope':'Fresh training/tuning/refit/calibration for three manuscript logistic blocks only; no new research experiment, no tree/window search, no historical chronology proof.','repository_root':str(root),'reference_root':str(ref),'results':results,'paired_bootstrap':paired,'bootstrap_replicates':ml.CFG['bootstrap'],'split_sizes':{name:{'n':len(part),'students':part.id_student.nunique()} for name,part in parts.items()},'source_and_input_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},'seed':ml.CFG['seed'],'runtime_seconds':time.time()-started,'all_match_within_1e_10':all(row['match_within_1e_10'] and row['bootstrap_interval_match_within_1e_10'] for row in results) and all(row['match_within_1e_10'] for row in paired)}
    (out/'core_retraining_verification.json').write_text(json.dumps(manifest,indent=2,default=int),encoding='utf8')
    if not manifest['all_match_within_1e_10']:
        warnings.warn('Retrained probabilities differ beyond tolerance: inspect manifest and preserve archived results.');return 2
    return 0

if __name__=='__main__': raise SystemExit(main())
