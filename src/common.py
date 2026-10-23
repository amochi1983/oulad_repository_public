# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/processed/matplotlib_cache'))
CFG=json.loads((ROOT/'configs/analysis.json').read_text())
KEY=['id_student','code_module','code_presentation']
for folder in ['data/processed','outputs/tables','outputs/figures','outputs/models','reports','tests']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
def table(name,frame):
    frame.to_csv(ROOT/f'outputs/tables/{name}.csv',index=False)
    (ROOT/f'outputs/tables/{name}.md').write_text(frame.to_markdown(index=False),encoding='utf-8')
def imd_group(value):
    if pd.isna(value):return 'Missing'
    lower=int(str(value).split('-')[0])
    return '0–30%' if lower<30 else ('30–70%' if lower<70 else '70–100%')
def eligible(frame,w):
    return frame.date_registration.notna() & (frame.date_registration<7*w) & (frame.date_unregistration.isna() | (frame.date_unregistration>=7*w))
def plot_save(name,frame,fig):
    frame.to_csv(ROOT/f'outputs/figures/{name}.csv',index=False)
    fig.savefig(ROOT/f'outputs/figures/{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(ROOT/f'outputs/figures/{name}.svg',bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def student_descriptive_bootstrap(df, masks, replicates=1000, seed=20261005):
    """Sample students globally; retain every enrolment with its cluster weight."""
    groups = pd.factorize(df.id_student)[0]
    n = len(np.unique(groups))
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(replicates):
        weights = np.bincount(rng.integers(0, n, n), minlength=n)[groups]
        row = {'replicate': b}
        for group, mask in masks.items():
            for field, column in [('success', 'success'), ('active', 'active_proportion')]:
                row[group+'_'+field] = (np.average(df[column].to_numpy()[mask], weights=weights[mask])
                    if weights[mask].sum() else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def named_wald(model, terms):
    """Explicit parameter-name constraints remain valid after formula reordering."""
    if not terms or any(t not in model.params.index for t in terms):
        raise ValueError('Wald test requires existing, nonempty named terms')
    return model.wald_test(', '.join(t+' = 0' for t in terms), scalar=True)


def checked_variance(value):
    value=float(value)
    if not np.isfinite(value) or value < -1e-12:
        raise ValueError(f'Invalid inferential variance: {value}')
    return max(0.0,value)

def validate_inference(model):
    if not model.converged or not np.isfinite(model.params).all() or not np.isfinite(model.cov_params()).all().all():
        raise ValueError('Nonconverged or nonfinite model inference')
    return model
