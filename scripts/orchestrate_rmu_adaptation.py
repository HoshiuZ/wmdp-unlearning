"""Run three minimum checks, three layer screens, two alpha screens, bounded fallback."""
import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PYTHON=ROOT/'envs/unlearning/bin/python'
def call(script,args,label):
 log=ROOT/'logs'/f'rmu_adaptation_{label}.console.log'
 if log.exists():raise RuntimeError(f'Existing log {log}; do not overwrite')
 print('START',label,flush=True)
 with log.open('w') as f:subprocess.run([str(PYTHON),'-u',str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',label,flush=True)
def select(args):return subprocess.check_output([str(PYTHON),str(ROOT/'scripts/finish_rmu_adaptation.py'),*args],cwd=ROOT,text=True).strip()
for hook in [7,12,18]:call('run_rmu_adaptation.py',['--hook',str(hook),'--minimal'],f'h{hook}_minimal')
for hook in [7,12,18]:call('run_rmu_adaptation.py',['--hook',str(hook)],f'h{hook}_screen40')
hook=select(['--select-layer']);print('SELECTED_LAYER',hook,flush=True)
for alpha in [2,4]:call('run_rmu_adaptation.py',['--hook',hook,'--alpha',str(alpha)],f'h{hook}_alpha{alpha}')
if select(['--need-fallback'])=='1':
 # Single-factor fallback around alpha1 layer configuration, not combined changes.
 call('run_rmu_adaptation.py',['--hook',hook,'--lr','2.5e-5'],f'h{hook}_lrhalf')
 call('run_rmu_adaptation.py',['--hook',hook,'--steering','1'],f'h{hook}_steering1')
call('finish_rmu_adaptation.py',['--full'],'selected_full')
print('ALL_COMPLETE',flush=True)
