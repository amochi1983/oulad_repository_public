# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

from pathlib import Path
import sys,subprocess,argparse,json,hashlib,time
ROOT=Path(__file__).resolve().parent

def step(args):
    p=subprocess.run([sys.executable,'-X','utf8',*args],cwd=ROOT)
    if p.returncode:raise RuntimeError(f'Failed step {args}, exit {p.returncode}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--rebuild-features',action='store_true');p.add_argument('--refit-association',action='store_true');p.add_argument('--retrain-core',action='store_true');p.add_argument('--post-audit',action='store_true');a=p.parse_args();start=time.time()
    if a.download:step(['src/download.py'])
    for item in json.loads((ROOT/'data/raw/manifest.json').read_text(encoding='utf8')):
        f=ROOT/'data/raw'/item['file']
        if not f.is_file():raise FileNotFoundError(f'{f}: use --download or place unchanged version1 files here')
        h=hashlib.sha256()
        with f.open('rb') as stream:
            while block:=stream.read(1024*1024):h.update(block)
        if h.hexdigest()!=item['sha256']:raise ValueError(f'Raw checksum mismatch {f.name}')
    if a.rebuild_features:
        for script in ['features','audit','descriptives']:step([f'src/{script}.py'])
    if a.refit_association:
        for script in ['association','first_project_statistics','manuscript_checks','first_project_figures']:step([f'src/{script}.py'])
    if a.post_audit or a.refit_association:step(['src/post_audit_checks.py'])
    if a.retrain_core:step(['tools/retrain_core.py','--repository-root',str(ROOT),'--reference-root',str(ROOT),'--output-dir',str(ROOT/'reports/core_retraining')])
    step(['-m','pytest','tests','-q','-p','no:cacheprovider'])
    status=dict(success=True,raw_hashes_valid=True,requested=vars(a),elapsed_seconds=time.time()-start,python=sys.version)
    (ROOT/'reports/reproduction_status.json').write_text(json.dumps(status,indent=2))
    mode='full_core_reproduction_status.json' if a.rebuild_features and a.refit_association and a.retrain_core else 'verification_status.json'
    (ROOT/'reports'/mode).write_text(json.dumps(status,indent=2))
    print('Requested steps and strict tests complete; see requested flags in reports/reproduction_status.json')

if __name__=='__main__':main()
