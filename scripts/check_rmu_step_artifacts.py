"""Final metadata/provenance check; no training, inference or historical rechecks."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'open-unlearning/saves/unlearn/rmu_step_comparison_v1/h12_a2_s32_seed42_continuous120'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
run=read(RUN/'run.json');assert run['status']=='completed' and run['reference_and_frozen_parameters_unchanged']
assert read(RUN/'trainer_state.json')['global_step']==120
rows=[json.loads(l) for l in (RUN/'training_metrics.jsonl').open()]
assert [r['step'] for r in rows if r['phase']=='train']==list(range(1,121))
assert [r['step'] for r in rows if r['phase']=='validation']==list(range(0,121,10))
assert len(run['callback_integrity'])==16 and all(x['weights_optimizer_scheduler_gradients_unchanged'] and x['rng_and_modes_restored'] for x in run['callback_integrity'])
for path,h in run['source_sha256'].items():assert sha(ROOT/path)==h,path
for path,h in read(RUN/'input_manifest_fingerprints.json')['files'].items():assert sha(ROOT/path)==h,path
baseline=read(ROOT/'open-unlearning/saves/unlearn/rmu_adaptation_v1/original_development_protocol_check/samples.json')
def identity(rows):return {(r['task'],r['doc_id'],r['doc_hash'],r['prompt_hash'],r['target']) for r in rows}
objects=[]
for step in [40,80,120]:
 cp=RUN/f'checkpoint-{step}';m=read(cp/'checkpoint.json');assert len(m['parameters'])==3
 check=read(cp/'rebuild_verification.json');assert check['compressed_payload_and_base_and_reconstructed_hashes_match']
 assert check['trained_parameter_hashes']=={r['name']:r['trained_tensor_sha256'] for r in m['parameters']}
 exposure=read(RUN/f'exposure_step{step}.json');assert all(len(r['indices'])==step*16 for r in exposure['roles'].values())
 dev=read(cp/'development/samples.json')
 for task,n in [('wmdp_bio',256),('mmlu',1710)]:assert len(dev[task])==n and identity(dev[task])==identity(baseline[task])
 full=read(RUN/f'full_evaluations/checkpoint-{step}/result.json')
 assert full['metrics']['wmdp_bio']['n']==1273 and full['metrics']['mmlu']['n']==14042
 for row in m['parameters']:
  p=cp/row['file'];assert subprocess.run(['git','check-ignore','-q',str(p.relative_to(ROOT))],cwd=ROOT).returncode==0
  objects.append(row['compressed_sha256'])
assert len(set(objects))==9
tracked=subprocess.check_output(['git','ls-files','--','open-unlearning/saves/unlearn/rmu_step_comparison_v1/**/*.xor.gz'],cwd=ROOT,text=True).strip();assert not tracked
assert not list(RUN.rglob('*.safetensors')) and not list(RUN.rglob('optimizer.pt'))
evalrun=read(RUN/'full_evaluations/evaluation_run.json');assert set(evalrun['steps'])=={'40','80','120'} and 'completed_at' in evalrun
out={'success':True,'training_updates':120,'validation_steps':list(range(0,121,10)),'checkpoint_steps':[40,80,120],'execution_source_hashes_match':True,'fixed_input_manifest_hashes_match':True,'callback_integrity_checks':16,'same_fixed_development_ids_and_targets':True,'full_sample_counts_verified':True,'new_delta_objects_local_only':9,'new_delta_pointers_tracked_by_git':False,'optimizer_state_or_full_model_copies_saved':False,'full_question_and_score_arithmetic_checks':'Performed inside evaluate_rmu_step_comparison.checked_summary for each of all three results','new_inference_or_training_started_by_this_check':False}
(ROOT/'logs/rmu_step_comparison_artifact_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
