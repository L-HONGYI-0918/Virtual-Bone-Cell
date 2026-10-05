# -*- coding: utf-8 -*-
""" (see paper) 135  (see paper)test (see paper): P/I/N  (see paper) 45  (see paper)"""
import sys, json, csv
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
ROOT=Path('G:/ (see paper)'); sys.path.insert(0,str(ROOT))
import numpy as np
from collections import defaultdict
pni={k:int(v) for k,v in json.loads((ROOT/'outputs'/'labels_expanded.json').read_text(encoding='utf-8')).items()}
with open(ROOT/'compoundinfo_beta.txt',encoding='utf-8') as f: rows=list(csv.DictReader(f,delimiter='\t'))
id2name={r['pert_id']:(r.get('cmap_name') or r.get('pert_id') or '').lower() for r in rows}
# start from large 25-each set
large=json.loads((ROOT/'outputs'/'test_set_drugs_large.json').read_text(encoding='utf-8'))
sel={0:list(large['promote_ids']),1:list(large['inhibit_ids']),2:list(large['neutral_ids'])}
sel_names={0:set(large['promote_names']),1:set(large['inhibit_names']),2:set(large['neutral_names'])}
# group remaining profiles by drug
by=defaultdict(lambda:{0:[],1:[],2:[]})
used=set(sel[0]+sel[1]+sel[2])
for p,y in pni.items():
    if p in used: continue
    by[id2name.get(p,p)][y].append(p)
rng=np.random.default_rng(20260814)
target=45
for y in [0,1,2]:
    candidates=[n for n in by if by[n][y] and n not in sel_names[y]]
    rng.shuffle(candidates)
    needed=target-len(sel_names[y])
    for n in candidates[:needed]:
        sel[y].extend(by[n][y]); sel_names[y].add(n)
    rng.shuffle(sel[y])
out={'promote_ids':sel[0],'inhibit_ids':sel[1],'neutral_ids':sel[2],
     'promote_names':sorted(sel_names[0]),'inhibit_names':sorted(sel_names[1]),'neutral_names':sorted(sel_names[2])}
path=ROOT/'outputs'/'test_set_drugs_135.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=1),encoding='utf-8')
print({k:(len(v) if 'ids' in k else len(v)) for k,v in out.items()})
print('saved',path)