"""Audit one fixed-seed replication; never select a different checkpoint."""
import argparse,json,csv,math,statistics,shutil
import torch
from rmu_adaptation_utils import ROOT,REPO,dump,materialized,digest,base_samples
from reweight_rmu_development import summarize,BIO_SUBJECTS
OUT=REPO/'saves/unlearn/simnpo_rmu_v1';OLD=OUT/'rmu_ce_s32_screen40';NEW=OUT/'rmu_ce_s32_seed43_screen40'
def read(p):return json.loads(p.read_text())
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--training-only',action='store_true');args=parser.parse_args()
 old=read(OLD/'run.json');run=read(NEW/'run.json');assert run['status']=='completed' and run['seed']==43 and run['max_steps']==40 and not run['gradient_cap']
 assert read(OUT/'rmu_ce_s32_seed43_minimal/run.json')['status']=='completed'
 cfg=read(NEW/'training_config.json');prev=read(OLD/'training_config.json')
 for k in ['model','data','collator','validation']:assert cfg[k]==prev[k]
 assert cfg['trainer']['method_args']==prev['trainer']['method_args'] and cfg['trainer']['handler']==prev['trainer']['handler']
 for k,v in cfg['trainer']['args'].items():
  if k not in ['output_dir','logging_dir','seed','data_seed']:assert v==prev['trainer']['args'][k]
 assert cfg['trainer']['args']['seed']==43 and cfg['trainer']['args']['data_seed']==43
 assert run['selected_parameters']==old['selected_parameters'] and run['frozen_parameters_unchanged'] and run['reference_unchanged']
 cv=read(NEW/'control_vector.json');oldcv=read(OLD/'control_vector.json');assert abs(cv['l2_norm']-32)<1e-4 and cv['values']!=oldcv['values']
 g=torch.Generator().manual_seed(43);v=torch.rand(1,1,2560,generator=g);expected=(v/v.norm()*32).flatten();assert torch.equal(torch.tensor(cv['values']),expected)
 manifest=read(ROOT/'logs/rmu_formal_blocks_manifest.json')
 for q in manifest['files']:assert digest((ROOT/q['path']).read_bytes())==q['sha256']
 consumed=read(NEW/'consumed_block_indices.json');oldconsumed=read(OLD/'consumed_block_indices.json');assert consumed!=oldconsumed
 budget={};input_tokens=0;loss_tokens=0
 for role in ['forget','retain']:
  blocks=[json.loads(l) for l in (ROOT/f'data/processed/bio_v1/formal_blocks_v1/{role}/train.jsonl').open()];ids=consumed[role];assert len(blocks)==2048 and len(ids)==640 and all(0<=i<len(blocks) for i in ids)
  count=sum(len(blocks[i]['input_ids']) for i in ids);input_tokens+=count;loss_tokens+=count-len(ids);budget[role]={'exposures':len(ids),'unique':len(set(ids)),'input_tokens':count,'effective_prediction_tokens':count-len(ids)}
 assert input_tokens==run['actual_input_tokens'] and loss_tokens==run['actual_loss_tokens']==run['effective_prediction_tokens']
 records=[json.loads(l) for l in (NEW/'training_metrics.jsonl').open()];train=[r for r in records if r['phase']=='train'];assert [r['step'] for r in train]==list(range(1,41));assert len(read(NEW/'component_metrics.json'))==640
 for role in consumed:assert [i for r in train for i in r['sampled_'+role+'_indices']]==consumed[role]
 size=0
 for c in run['development']:
  assert c['save_reload_identical'];cp=ROOT/c['checkpoint'];m=read(cp/'checkpoint.json');assert len(m['parameters'])==3
  for q in m['parameters']:
   obj=materialized(cp/q['file']);assert digest(obj.read_bytes())==q['compressed_sha256'] and obj.stat().st_size==q['bytes'];size+=q['bytes']
 ref=read(OUT.parent/'rmu_adaptation_v1/original_development_protocol_check/samples.json');baseline=summarize(ref['mmlu']);dev=[]
 for step in [10,20,40]:
  path=NEW/f'checkpoint-{step}/development/samples.json';data=read(path)
  for task,n in [('wmdp_bio',256),('mmlu',1710)]:
   base={(x['task'],x['doc_id']):x for x in ref[task]};assert len(data[task])==n and {(x['task'],x['doc_id']) for x in data[task]}==set(base)
   for x in data[task]:
    y=base[x['task'],x['doc_id']];assert all(x[k]==y[k] for k in ['doc_hash','prompt_hash','target'])
  m=summarize(data['mmlu']);dev.append({'seed':43,'step':step,'bio_percent':100*sum(x['acc'] for x in data['wmdp_bio'])/256,'mmlu':m,'weighted_change_pp':m['all']['weighted_percent']-baseline['all']['weighted_percent'],'path':str(path.relative_to(ROOT))})
 dump(ROOT/'logs/rmu_ce_seed43_development.json',{'rows':dev,'used_for_selection':False,'fixed_full_checkpoint':40})
 audit={'seed':43,'training_config_only_seed_changed':True,'control_direction_regenerated':True,'control_reproduced_from_seed43':True,'old_control_sha256':oldcv['sha256'],'new_control_sha256':cv['sha256'],'sampling_changed':True,'data_manifest_sha_verified':True,'budget':budget,'actual_input_tokens':input_tokens,'actual_loss_tokens':loss_tokens,'input_token_change_from_seed42':input_tokens-old['actual_input_tokens'],'old_input_tokens':old['actual_input_tokens'],'old_loss_tokens':old['actual_loss_tokens'],'selected_parameters':run['selected_parameters'],'frozen_reference_unchanged':True,'reload_and_real_lfs_sha_verified':True,'development_requests_verified':True,'compressed_bytes':size,'training_code_commit':run['code_commit'],'training_seconds':run['elapsed_seconds'],'cuda_peak_allocated_bytes':run['cuda_peak_allocated_bytes'],'cuda_peak_reserved_bytes':run['cuda_peak_reserved_bytes'],'space':shutil.disk_usage(ROOT)._asdict()}
 dump(ROOT/'logs/rmu_ce_seed43_training_audit.json',audit);print('TRAIN_AUDIT',input_tokens,loss_tokens,dev[-1]['bio_percent'],dev[-1]['weighted_change_pp'],flush=True)
 if args.training_only:return
 dest=OUT/'seed43_full_evaluations/rmu_ce_s32_seed43_screen40_checkpoint-40';result=read(dest/'result.json');data=read(dest/'samples.json')
 for task,subjects in base_samples().items():
  base={(name,x['doc_id']):x for name,rs in subjects.items() for x in rs};assert len(data[task])==len(base) and {(x['task'],x['doc_id']) for x in data[task]}==set(base)
  for x in data[task]:
   y=base[x['task'],x['doc_id']];assert x['doc_hash']==y['doc_hash'] and x['prompt_hash']==y['prompt_hash'] and x['target']==int(y['doc']['answer']) and x['baseline_acc']==int(y['acc'])
 oldresult=read(OUT/'s32_full_evaluations/rmu_ce_s32_screen40_checkpoint-40/result.json');fullrows=[]
 for seed,r in [(42,oldresult),(43,result)]:
  m=r['metrics'];fullrows.append({'seed':seed,'hook':12,'parameters':'10/11/12 down_proj.weight','lr':5e-5,'alpha':2,'steering':32,'ce_weight':run['coefficients']['ce_weight'],'steps':40,'bio_percent':m['wmdp_bio']['accuracy_percent'],'bio_change_pp':m['wmdp_bio']['change_percentage_points'],'mmlu_percent':m['mmlu']['accuracy_percent'],'mmlu_change_pp':m['mmlu']['change_percentage_points'],'meets_2pp':m['mmlu']['change_percentage_points']>=-2})
 base_mmlu=[{'task':name,'acc':int(x['acc'])} for name,records in base_samples()['mmlu'].items() for x in records];olddata=read(OUT/'s32_full_evaluations/rmu_ce_s32_screen40_checkpoint-40/samples.json')
 groups={'baseline':summarize(base_mmlu),'seed42':summarize(olddata['mmlu']),'seed43':summarize(data['mmlu'])}
 for name in ['seed42','seed43']:
  for group in ['all','biology_related','other']:groups[name][group]['change_pp']=groups[name][group]['weighted_percent']-groups['baseline'][group]['weighted_percent']
 correct_losses={g:sum(groups['seed42']['per_subject'][t]['correct']-groups['seed43']['per_subject'][t]['correct'] for t in names) for g,names in [('biology_related',BIO_SUBJECTS),('other',set(groups['baseline']['per_subject'])-BIO_SUBJECTS)]};assert sum(correct_losses.values())==oldresult['metrics']['mmlu']['correct']-result['metrics']['mmlu']['correct']
 subject_rows=[]
 for task,b in groups['baseline']['per_subject'].items():
  a42=groups['seed42']['per_subject'][task];a43=groups['seed43']['per_subject'][task];subject_rows.append({'task':task,'n':b['full_weight_n'],'baseline_percent':b['accuracy_percent'],'seed42_percent':a42['accuracy_percent'],'seed43_percent':a43['accuracy_percent'],'seed42_change_pp':a42['accuracy_percent']-b['accuracy_percent'],'seed43_change_pp':a43['accuracy_percent']-b['accuracy_percent'],'seed43_minus_seed42_pp':a43['accuracy_percent']-a42['accuracy_percent']})
 dump(ROOT/'logs/rmu_ce_seed43_full_subjects.json',{'groups':groups,'subjects':subject_rows,'correct_loss_43_vs_42':correct_losses,'all_full_questions':True})
 with (ROOT/'logs/rmu_ce_seed43_full_subjects.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(subject_rows[0]),lineterminator='\n');w.writeheader();w.writerows(subject_rows)
 stats={k:{'mean':statistics.mean(r[k] for r in fullrows),'sample_sd':statistics.stdev(r[k] for r in fullrows)} for k in ['bio_percent','mmlu_percent','mmlu_change_pp']};summary={'rows':fullrows,'n_seeds':2,'statistics':stats,'all_seeds_meet_2pp':all(r['meets_2pp'] for r in fullrows),'fixed_step':40,'exploratory':True,'no_development_selection':True};dump(ROOT/'logs/rmu_ce_seed43_full_comparison.json',summary)
 with (ROOT/'logs/rmu_ce_seed43_full_comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(fullrows[0]),lineterminator='\n');w.writeheader();w.writerows(fullrows)
 audit.update(full_requests_verified=True,full_counts={k:len(v) for k,v in data.items()},all_seeds_meet_2pp=summary['all_seeds_meet_2pp'],full_result=str((dest/'result.json').relative_to(ROOT)));dump(ROOT/'logs/rmu_ce_seed43_completion_audit.json',audit)
 lines=['# RMU＋CE seed43复验结果','','本轮固定同一配置，只新增seed43，预先固定全量评测第40步，不按开发结果挑选。数据文章划分和候选块不变；训练seed、data_seed改为43，控制方向原生随机生成后固定。','', '| seed | 步 | Bio% | Bio较原始变化点 | MMLU% | MMLU较原始变化点 | 满足2点目标 |','| --- | ---: | ---: | ---: | ---: | ---: | --- |']
 for r in fullrows:lines.append(f"| {r['seed']} | 40 | {r['bio_percent']:.4f} | {r['bio_change_pp']:+.4f} | {r['mmlu_percent']:.4f} | {r['mmlu_change_pp']:+.4f} | {'是' if r['meets_2pp'] else '否'} |")
 lines+=['',f"两次均值：Bio {stats['bio_percent']['mean']:.4f}%、MMLU {stats['mmlu_percent']['mean']:.4f}%、MMLU变化 {stats['mmlu_change_pp']['mean']:+.4f}点。样本标准差分别为{stats['bio_percent']['sample_sd']:.4f}、{stats['mmlu_percent']['sample_sd']:.4f}、{stats['mmlu_change_pp']['sample_sd']:.4f}点。仅两次运行，均值不能代替单次约束判断。seed42为前期筛选所得，这些均值和标准差只作描述。",'']
 if summary['all_seeds_meet_2pp']:lines+=['seed42与43均达到本轮2点目标，复验支持该配置在这两次运行中的取舍；样本量仍不足以证明对seed稳定。下一步建议先固定独立验证方案，再决定是否对该配置补足第三个seed和匹配的RMU对照，不继续系数搜索。本轮不自动追加。']
 else:lines+=['至少一个seed未达到2点目标，本配置尚不能作为稳定满足约束的方案；不以两次均值掩盖失败。保留seed42已观察结果，并降低对其稳定性的判断。下一步建议只把CE保留权重提高到当前的1.5倍，并固定seed42/43进行匹配复验，其余参数不变；本轮不自动追加。']
 lines+=['','固定学科组全量结果（沿用既有10个生物相关/47个其他学科定义）：','', '| 组 | 题数 | 原始% | seed42% | seed43% | seed43较原始变化点 |','| --- | ---: | ---: | ---: | ---: | ---: |']
 for group in ['biology_related','other']:
  b=groups['baseline'][group];g42=groups['seed42'][group];g43=groups['seed43'][group];lines.append(f"| {group} | {b['full_test_questions']} | {b['weighted_percent']:.4f} | {g42['weighted_percent']:.4f} | {g43['weighted_percent']:.4f} | {g43['change_pp']:+.4f} |")
 lines+=['',f"seed43较seed42的MMLU少答对{sum(correct_losses.values())}题，其中生物相关组{correct_losses['biology_related']}题、其他学科{correct_losses['other']}题；正常能力损失分布在两类学科。"]
 lines+=['','各科完整变化保存在logs/rmu_ce_seed43_full_subjects.csv；这些是固定全量预测的描述性分解，未用于修改本轮配置。']
 lines+=['','原始全量Bio73.8413%、MMLU70.6737%；同一归档0shot题目和loglikelihood请求，BF16/SDPA、batch1、无聊天模板。官方test已参与前期探索，本轮是固定配置seed复验，不是新的独立测试。','', '## 训练与检查','',f"hook12输出，10/11/12 down_proj.weight共74,711,040参数，lr5e-5、alpha2、steering32、CE权重{run['coefficients']['ce_weight']}、batch1/累积16、40步。每角色640次输入；seed43 input/loss/prediction={input_tokens}/{loss_tokens}/{loss_tokens}，seed42为320619/319339/319339。采样改变使输入token相差{input_tokens-old['actual_input_tokens']:+d}，本轮匹配更新次数和样本曝光量，没有声称精确匹配token。独立按缓存ID长度重新计算通过。",'',f"控制向量norm={cv['l2_norm']:.6f}，SHA {cv['sha256']}，与seed42不同。随机生成规则未变，只由seed43控制；最小检查与正式运行各自从原始开始。冻结参数和参考模型不变，3增量重建逐位相同及真实LFS对象SHA通过，合计{size/2**20:.2f}MiB。无optimizer状态，不能断点续训。",'',f"训练{run['elapsed_seconds']:.1f}秒（含验证与开发回调），峰值allocated {run['cuda_peak_allocated_bytes']/2**30:.3f}GiB/reserved {run['cuda_peak_reserved_bytes']/2**30:.3f}GiB；代码{run['code_commit']}；数据SHA及来源见logs/rmu_formal_blocks_manifest.json。完成时余{audit['space']['free']/2**30:.2f}GiB。",'','## 开发记录（未用于重选）','', '| 步 | Bio% | MMLU加权% | 加权MMLU变化点 |','| --- | ---: | ---: | ---: |']
 for r in dev:lines.append(f"| {r['step']} | {r['bio_percent']:.4f} | {r['mmlu']['all']['weighted_percent']:.4f} | {r['weighted_change_pp']:+.4f} |")
 lines+=['','完整配置、数据SHA、样本ID、控制向量、损失、逐题预测、代码与检查结果已保存；所有旧结果保留，仅本地Git提交，未自动推送。']
 (ROOT/'docs/rmu-ce-seed43-results.md').write_text('\n'.join(lines)+'\n');print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
