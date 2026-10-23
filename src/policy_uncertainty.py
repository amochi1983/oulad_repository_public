# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

import numpy as np,pandas as pd,json
from common import *
from ml import measure,bootstrap
def run():
    d=pd.read_parquet(ROOT/'outputs/models/predictions.parquet');d=d[d.split=='2014_test'];rows=[];sub=[];selected=json.loads((ROOT/'reports/selected_boosting.json').read_text())['model']
    for (w,block,name),a in d.groupby(['week','block','model']):
        y=a.risk.to_numpy();p=a.probability.to_numpy();ids=a.id_student.to_numpy()
        for policy in ['0.5','validation_F1','top10','top20']:
            t=.5 if policy=='0.5' else a.threshold.iloc[0];flag=(p>=t).astype(int)
            if policy.startswith('top'):
                flag[:]=0;flag[np.argsort(-p,kind='stable')[:int(np.ceil(len(p)*int(policy[3:])/100))]]=1
            b=bootstrap(y,p,ids,alert=flag);r=dict(week=w,block=block,model=name,policy=policy,n=len(y),alerts=int(flag.sum()),alert_rate=flag.mean(),**measure(y,p,alert=flag))
            for metric in b:r[metric+'_low'],r[metric+'_high']=b[metric].quantile([.025,.975])
            rows.append(r)
            if w==4 and block=='D' and name==selected:
                for g,group in a.groupby('imd_group'):
                    ix=a.imd_group.eq(g).to_numpy();bb=bootstrap(y[ix],p[ix],ids[ix],alert=flag[ix]);r=dict(week=w,model=name,policy=policy,imd_group=g,n=int(ix.sum()),prevalence=y[ix].mean(),alert_rate=flag[ix].mean(),**measure(y[ix],p[ix],alert=flag[ix]))
                    for metric in bb:r[metric+'_low'],r[metric+'_high']=bb[metric].quantile([.025,.975])
                    sub.append(r)
    table('T8_alert_policies',pd.DataFrame(rows));table('T8_alert_policy_subgroups',pd.DataFrame(sub))
if __name__=='__main__':run()
