"""Two authorized single-factor extensions after phase1 full constraint failure."""
import subprocess,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PYTHON=ROOT/'envs/unlearning/bin/python'
assert not json.loads((ROOT/'logs/rmu_adaptation_result.json').read_text())['full_mmlu_constraint_pass']
assert shutil.disk_usage(ROOT).free>=1024**3,'Wait for expanded data filesystem: require >=1GiB available before extensions'
def call(script,args,label):
 log=ROOT/'logs'/f'rmu_adaptation_{label}.console.log'
 if log.exists():raise RuntimeError(f'Refuse to overwrite {log}')
 print('START',label,flush=True)
 with log.open('w') as f:subprocess.run([str(PYTHON),'-u',str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',label,flush=True)
call('run_rmu_adaptation.py',['--hook','12','--lr','2.5e-5'],'h12_lrhalf')
call('run_rmu_adaptation.py',['--hook','12','--steering','1'],'h12_steering1')
call('finish_rmu_adaptation.py',['--phase2','--full'],'selected_full_phase2')
print('PHASE2_COMPLETE',flush=True)
