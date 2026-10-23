# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

"""Prespecified secondary outcome and assessment sensitivities at week four."""
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,accuracy_score,log_loss
from sklearn.linear_model import LogisticRegression
from common import *
from ml import pipeline,split2013,columns,measure,bootstrap
def run():
    df=pd.read_parquet(ROOT/'data/processed/week_4.parquet');records=[];preds=[]
    specs=[('subsequent_withdrawal','withdrawal',df,[]),('conditional_fail','risk',df[df.final_result!='Withdrawn'],[]),('submission_sensitivity','risk',df,['observed_submissions']),('score_availability_sensitivity','risk',df,['observed_submissions','score_sensitivity']),('consistent_withdrawal_dates','risk',df[~df.inconsistent_withdrawal],[])]
    for name,outcome,d,extra in specs:
        test=d[d.year==2014];train=d[(d.year==2013)&~d.id_student.isin(test.id_student)];f,v,c=split2013(train)
        m=pipeline(f,'D','Logistic',1)
        if extra:
            transformers=m.named_steps['preprocess'].transformers
            transformers[1]=(transformers[1][0],transformers[1][1],transformers[1][2]+extra)
        fitdata=pd.concat([f,v]);m.fit(fitdata,fitdata[outcome]);p=m.predict_proba(test)[:,1];y=test[outcome].to_numpy();b=bootstrap(y,p,test.id_student.to_numpy());r=dict(analysis=name,positive_class='withdrawal' if outcome=='withdrawal' else 'non-success',n=len(test),train_n=len(train),prevalence=y.mean(),score_release_dates_known=False if 'score' in name else None,**measure(y,p))
        for metric in b:r[metric+'_low'],r[metric+'_high']=b[metric].quantile([.025,.975])
        records.append(r);out=test[KEY].copy();out['outcome']=y;out['probability']=p;out['analysis']=name;preds.append(out)
    test=df[df.year==2014];train=df[(df.year==2013)&~df.id_student.isin(test.id_student)];f,v,c=split2013(train)
    if f.final_result.value_counts().min()>=100:
        m=pipeline(f,'D','Logistic',1);fitdata=pd.concat([f,v]);m.fit(fitdata,fitdata.final_result);p=m.predict_proba(test);yp=m.predict(test);classes=m.classes_;records.append(dict(analysis='four_class_exploratory',n=len(test),train_n=len(fitdata),accuracy=accuracy_score(test.final_result,yp),macro_F1=f1_score(test.final_result,yp,average='macro'),log_loss=log_loss(test.final_result,p,labels=classes),classes=','.join(classes)))
        out=test[KEY+['final_result']].copy()
        for i,label in enumerate(classes):out['prob_'+label]=p[:,i]
        out.to_csv(ROOT/'outputs/models/four_class_predictions.csv',index=False)
    table('T8_secondary_sensitivities',pd.DataFrame(records));pd.concat(preds).to_csv(ROOT/'outputs/models/secondary_predictions.csv',index=False)
if __name__=='__main__':run()
