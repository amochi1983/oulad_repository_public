# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Aymen Ochi
# Author: Aymen Ochi

"""Resolve official metadata; validate archive and retain unchanged CSV bytes."""
import hashlib,json,time,zipfile,urllib.request
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
FILES=['courses','studentInfo','studentRegistration','studentVle','vle','assessments','studentAssessment']
def fetch(url,path):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url,timeout=120) as r, path.open('wb') as out:
                while chunk:=r.read(1024*1024): out.write(chunk)
            return
        except Exception:
            if attempt==3: raise
            time.sleep(2**attempt)
def digest(path,kind='sha256'):
    h=hashlib.new(kind)
    with path.open('rb') as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()
def main():
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    meta=raw/'figshare_metadata.json'
    if not meta.exists(): fetch('https://api.figshare.com/v2/articles/5081998/versions/1',meta)
    d=json.loads(meta.read_text());f=next(x for x in d['files'] if x['name'].endswith('.zip'))
    archive=raw/f['name']
    if not archive.exists():
        tmp=archive.with_suffix('.part');fetch(f['download_url'],tmp);tmp.replace(archive)
    assert archive.stat().st_size==f['size'],'Archive size differs from official metadata'
    assert digest(archive,'md5')==f['computed_md5'],'Official archive checksum mismatch'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for info in z.infolist():
            target=(raw/info.filename).resolve()
            assert target.is_relative_to(raw.resolve()),'Unsafe archive member'
            if Path(info.filename).name in [x+'.csv' for x in FILES]:
                target=raw/Path(info.filename).name
                if not target.exists(): target.write_bytes(z.read(info))
    previous={x['file']:x for x in json.loads((raw/'manifest.json').read_text())} if (raw/'manifest.json').exists() else {}
    manifest=[]
    for p in [archive]+[raw/(x+'.csv') for x in FILES]:
        manifest.append(dict(file=p.name,source_url=f['download_url'],metadata_url=d['url'],download_date=previous.get(p.name,{}).get('download_date',datetime.now(timezone.utc).isoformat()),size=p.stat().st_size,sha256=digest(p),license=d['license'],citation=d['citation']))
    # Preserve the shipped source-of-truth hashes: do not legitimise altered files
    # by rewriting their expected manifest after acquisition.
    for item in manifest:
        expected=previous.get(item['file'])
        if expected and item['sha256'] != expected['sha256']:
            raise ValueError(f"Pinned raw SHA256 mismatch: {item['file']}")
    if previous:
        (raw/'last_acquisition.json').write_text(json.dumps(manifest,indent=2))
    else:
        (raw/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Official archive and seven raw CSVs verified',flush=True)
if __name__=='__main__':main()
