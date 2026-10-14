"""Audit fixed CE1.5 matched trials and report both seeds without reselection."""
import argparse,json,csv,statistics,shutil
from rmu_adaptation_utils import ROOT,REPO,dump,materialized,digest,base_samples
from reweight_rmu_development import summarize
OUT=REPO/'saves/unlearn/simnpo_rmu_v1'
def read(p):return json.loads(p.read_text())
def name(seed,scale=False):return 'rmu_ce_s32'+('_seed43' if seed==43 else '')+('_ce15' if scale else '')+'_screen40'
def full_dest(seed,scale=False):
 family='ce15_full_evaluations' if scale else ('s32_full_evaluations' if seed==42 else 'seed43_full_evaluations')
 return OUT/family/(name(seed,scale)+'_checkpoint-40')
def verify_requests(data,reference):
 for task,base in reference.items():
  rows=data[task];assert len(rows)==len(base) and {(x['task'],x['doc_id']) for x in rows}==set(base)
  for x in rows:
   y=base[x['task'],x['doc_id']];assert all(x[k]==y[k] for k in ['doc_hash','prompt_hash','target','baseline_acc'])
def dev_reference():
 d=read(OUT.parent/'rmu_adaptation_v1/original_development_protocol_check/samples.json');return {task:{(x['task'],x['doc_id']):x for x in rows} for task,rows in d.items()}
def audit(seed):
 trial=OUT/name(seed,True);oldtrial=OUT/name(seed);run=read(trial/'run.json');old=read(oldtrial/'run.json');assert run['status']=='completed' and run['ce_scale']==1.5 and run['seed']==seed and run['max_steps']==40
 minimal=read(OUT/name(seed,True).replace('_screen40','_minimal')/'run.json');assert minimal['status']=='completed' and len(minimal['gradient_checks'])==2
 cfg=read(trial/'training_config.json');prior=read(oldtrial/'training_config.json')
 for k in ['model','data','collator','validation']:assert cfg[k]==prior[k]
 for k,v in cfg['trainer']['args'].items():
  if k not in ['output_dir','logging_dir']:assert v==prior['trainer']['args'][k]
 method=dict(cfg['trainer']['method_args']);assert method.pop('ce_weight')==prior['trainer']['method_args']['ce_weight']*1.5
 before=dict(prior['trainer']['method_args']);before.pop('ce_weight');assert method==before
 assert run['selected_parameters']==old['selected_parameters'] and run['frozen_parameters_unchanged'] and run['reference_unchanged']
 assert read(trial/'control_vector.json')==read(oldtrial/'control_vector.json')
 consumed=read(trial/'consumed_block_indices.json');assert consumed==read(oldtrial/'consumed_block_indices.json')
 input_tokens=0;loss_tokens=0;budgets={}
 for role in ['forget','retain']:
  blocks=[json.loads(l) for l in (ROOT/f'data/processed/bio_v1/formal_blocks_v1/{role}/train.jsonl').open()];ids=consumed[role];assert len(blocks)==2048 and len(ids)==640
  count=sum(len(blocks[i]['input_ids']) for i in ids);input_tokens+=count;loss_tokens+=count-len(ids);budgets[role]={'exposures':len(ids),'unique':len(set(ids)),'input_tokens':count}
 assert input_tokens==run['actual_input_tokens']==old['actual_input_tokens'];assert loss_tokens==run['actual_loss_tokens']==run['effective_prediction_tokens']==old['effective_prediction_tokens']
 rows=[json.loads(l) for l in (trial/'training_metrics.jsonl').open()];train=[r for r in rows if r['phase']=='train'];assert [r['step'] for r in train]==list(range(1,41)) and len(read(trial/'component_metrics.json'))==640
 for role in consumed:assert [i for r in train for i in r['sampled_'+role+'_indices']]==consumed[role]
 size=0
 for c in run['development']:
  assert c['save_reload_identical'];cp=ROOT/c['checkpoint'];m=read(cp/'checkpoint.json');assert len(m['parameters'])==3
  for q in m['parameters']:
   obj=materialized(cp/q['file']);assert digest(obj.read_bytes())==q['compressed_sha256'] and obj.stat().st_size==q['bytes'];size+=q['bytes']
 ref=dev_reference();baseline=summarize(list(ref['mmlu'].values()));dev=[]
 for step in [10,20,40]:
  path=trial/f'checkpoint-{step}/development/samples.json';data=read(path);verify_requests(data,ref);m=summarize(data['mmlu']);dev.append({'seed':seed,'step':step,'bio_percent':100*sum(r['acc'] for r in data['wmdp_bio'])/256,'mmlu':m,'weighted_change_pp':m['all']['weighted_percent']-baseline['all']['weighted_percent']})
 result={'seed':seed,'only_ce_weight_changed':True,'coefficient':run['coefficients']['ce_weight'],'control_sampling_and_tokens_exactly_matched':True,'budget':budgets,'input_tokens':input_tokens,'loss_tokens':loss_tokens,'frozen_reference_unchanged':True,'reload_and_real_lfs_sha_verified':True,'development_requests_verified':True,'compressed_bytes':size,'training_code_commit':run['code_commit'],'script_sha256':run['script_sha256'],'seconds':run['elapsed_seconds'],'cuda_peak_allocated_bytes':run['cuda_peak_allocated_bytes'],'cuda_peak_reserved_bytes':run['cuda_peak_reserved_bytes'],'development':dev}
 dump(ROOT/f'logs/rmu_ce15_seed{seed}_training_audit.json',result);print('TRAIN_AUDIT',seed,dev[-1]['bio_percent'],dev[-1]['weighted_change_pp'],flush=True);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,43]);p.add_argument('--training-only',action='store_true');a=p.parse_args();seeds=[a.seed] if a.seed else [42,43]
 for q in read(ROOT/'logs/rmu_formal_blocks_manifest.json')['files']:assert digest((ROOT/q['path']).read_bytes())==q['sha256']
 audits=[audit(seed) for seed in seeds]
 if a.training_only:return
 assert seeds==[42,43] and audits[0]['script_sha256']==audits[1]['script_sha256'];reference={task:{(s,x['doc_id']):{'doc_hash':x['doc_hash'],'prompt_hash':x['prompt_hash'],'target':int(x['doc']['answer']),'baseline_acc':int(x['acc'])} for s,rs in subjects.items() for x in rs} for task,subjects in base_samples().items()}
 rows=[];groups={};paired=[]
 for seed in seeds:
  for scale in [False,True]:
   dest=full_dest(seed,scale);result=read(dest/'result.json');data=read(dest/'samples.json');verify_requests(data,reference);m=result['metrics'];assert all(m[t]['n']==len(rs) and m[t]['correct']==sum(x['acc'] for x in rs) and abs(m[t]['accuracy_percent']-100*sum(x['acc'] for x in rs)/len(rs))<1e-9 for t,rs in data.items());s=summarize(data['mmlu']);assert abs(s['all']['weighted_percent']-m['mmlu']['accuracy_percent'])<1e-9
   row={'seed':seed,'ce_scale':1.5 if scale else 1.,'hook':12,'parameters':'10/11/12 down_proj.weight','lr':5e-5,'alpha':2,'steering':32,'ce_weight':audits[seeds.index(seed)]['coefficient'] if scale else audits[seeds.index(seed)]['coefficient']/1.5,'steps':40,'bio_percent':m['wmdp_bio']['accuracy_percent'],'bio_change_pp':m['wmdp_bio']['change_percentage_points'],'mmlu_percent':m['mmlu']['accuracy_percent'],'mmlu_change_pp':m['mmlu']['change_percentage_points'],'meets_2pp':m['mmlu']['change_percentage_points']>=-2,'result':str((dest/'result.json').relative_to(ROOT))};rows.append(row);groups[f'seed{seed}_ce{row["ce_scale"]}']=s
  old,new=rows[-2:];paired.append({'seed':seed,'bio_change_ce15_minus_ce1':new['bio_percent']-old['bio_percent'],'mmlu_change_ce15_minus_ce1':new['mmlu_percent']-old['mmlu_percent']})
 stats={str(scale):{k:{'mean':statistics.mean(r[k] for r in rows if r['ce_scale']==scale),'sample_sd':statistics.stdev(r[k] for r in rows if r['ce_scale']==scale)} for k in ['bio_percent','mmlu_percent','mmlu_change_pp']} for scale in [1.,1.5]};passed=all(r['meets_2pp'] for r in rows if r['ce_scale']==1.5)
 dump(ROOT/'logs/rmu_ce15_full_comparison.json',{'rows':rows,'paired':paired,'statistics':stats,'ce15_both_seeds_meet_2pp':passed,'fixed_step':40,'used_development_for_selection':False,'groups':groups})
 with (ROOT/'logs/rmu_ce15_full_comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 dump(ROOT/'logs/rmu_ce15_completion_audit.json',{'training':audits,'full_requests_verified':True,'data_manifest_sha_verified':True,'full_evaluations':2,'ce15_both_seeds_meet_2pp':passed,'space':shutil.disk_usage(ROOT)._asdict()})
 lines=['# CE保留权重1.5倍匹配对照结果','','仅改变CE保留权重，seed42/43各自与旧CE1倍匹配。两组均独立从原始模型开始，预先固定第40步全量评测。','', '| seed | CE倍率 | Bio% | MMLU% | MMLU较原始变化点 | 满足2点目标 |','| --- | ---: | ---: | ---: | ---: | --- |']
 for r in rows:lines.append(f"| {r['seed']} | {r['ce_scale']} | {r['bio_percent']:.4f} | {r['mmlu_percent']:.4f} | {r['mmlu_change_pp']:+.4f} | {'是' if r['meets_2pp'] else '否'} |")
 lines+=['','同seed内CE1.5减CE1：','', '| seed | Bio变化点 | MMLU变化点 |','| --- | ---: | ---: |']
 for r in paired:lines.append(f"| {r['seed']} | {r['bio_change_ce15_minus_ce1']:+.4f} | {r['mmlu_change_ce15_minus_ce1']:+.4f} |")
 lines+=['', '两seed的CE1.5结果'+('均满足' if passed else '未全部满足')+'本轮2点目标；逐seed判断优先于均值。','']
 for scale in [1.,1.5]:
  st=stats[str(scale)];lines.append(f"CE{scale}两次描述性均值：Bio{st['bio_percent']['mean']:.4f}%（样本SD {st['bio_percent']['sample_sd']:.4f}点）、MMLU{st['mmlu_percent']['mean']:.4f}%（SD {st['mmlu_percent']['sample_sd']:.4f}点），MMLU变化{st['mmlu_change_pp']['mean']:+.4f}点。")
 lines+=['','seed42旧配置来自前期筛选，只有两个seed，统计仅作描述。测试集已参与前期探索，本轮是固定方案匹配对照；开发指标未用于改变检查点或配置。原始全量Bio73.8413%、MMLU70.6737%；Bio1273/MMLU14042，同一归档0shot题目、提示、答案与loglikelihood评分，BF16/SDPA、batch1、无聊天模板。','','## 配置和保存','','hook12输出（0-based），10/11/12 down_proj.weight共74,711,040参数，lr5e-5、alpha2、steering32、40步、batch1/累积16、256token上限，CE0.060028908617563034→0.09004336292634455。无Sim遗忘项。不改变文章划分、数据版本、优化器或其他损失权重。']
 for au in audits:lines+=['',f"seed{au['seed']}：每角色640次输入，input/loss token={au['input_tokens']}/{au['loss_tokens']}，与该seed旧实验采样/方向/预算逐值一致。训练{au['seconds']:.1f}秒（含回调），峰值allocated {au['cuda_peak_allocated_bytes']/2**30:.3f}GiB/reserved {au['cuda_peak_reserved_bytes']/2**30:.3f}GiB；代码{au['training_code_commit']}，脚本SHA {au['script_sha256']}；3增量{au['compressed_bytes']/2**20:.2f}MiB。"]
 lines+=['','两组冻结参数和参考模型不变；最小CUDA检查、实际缓存token独立核算、逐位重建及真实LFS对象SHA、开发/全量请求逐题核对通过。6增量及所有旧结果保留；无optimizer状态，不能断点续训。数据SHA见logs/rmu_formal_blocks_manifest.json，完整配置/采样/控制向量/分项loss/逐题预测均保存。','','本轮结束后按用户授权推送代码、模型增量与结果到GitHub；私人根目录文档和受限正文/缓存继续排除。']
 (ROOT/'docs/rmu-ce15-results.md').write_text('\n'.join(lines)+'\n');print('COMPLETE',json.dumps({'rows':rows,'paired':paired,'both_pass':passed},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
