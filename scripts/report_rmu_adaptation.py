"""Audit completed adaptation artifacts and emit Chinese comparison report."""
import json,csv,math
from pathlib import Path
from rmu_adaptation_utils import ROOT,REPO,OUT,dump,materialized,digest,base_samples
from finish_rmu_adaptation import candidates

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--phase2',action='store_true');args=parser.parse_args();suffix='_phase2' if args.phase2 else ''
 result=json.loads((ROOT/f'logs/rmu_adaptation_result{suffix}.json').read_text());rows=candidates();runs=[]
 import shutil
 shutil.copyfile(ROOT/'models/Qwen3-4B-Instruct-2507/LICENSE',OUT/'LICENSE')
 dump(OUT/'selected_checkpoint.json',result['selection']['selected'])
 for p in OUT.glob('*_screen40/run.json'):
  x=json.loads(p.read_text());assert x['status']=='completed';runs.append(x)
 reference=json.loads((OUT/'h7_a1_lr5e-05_s2_screen40/consumed_block_indices.json').read_text())
 control=json.loads((OUT/'h7_a1_lr5e-05_s2_screen40/control_vector.json').read_text())
 for x in runs:
  p=OUT/x['name'];assert json.loads((p/'consumed_block_indices.json').read_text())==reference,'cross-run sampling differs'
  c=json.loads((p/'control_vector.json').read_text());assert all(abs(v/x['steering_coefficient']-w/2)<1e-7 for v,w in zip(c['values'],control['values']))
  assert x['frozen_parameters_unchanged'] and x['reference_fingerprints_unchanged']
  assert all(d['save_reload_identical'] for d in x['development'])
  for m in p.glob('checkpoint-*/checkpoint.json'):
   for r in json.loads(m.read_text())['parameters']:
    file=materialized(m.parent/r['file']);assert digest(file.read_bytes())==r['compressed_sha256']
 current=OUT/'h7_a1_lr5e-05_s2_screen40/training_metrics.jsonl';old=REPO/'saves/unlearn/qwen3_4b_rmu_bio_formal80_v1/training_metrics.jsonl'
 def train(p):return [json.loads(l) for l in p.open() if json.loads(l)['phase']=='train']
 aa=train(current);bb=train(old)[:40];loss_diff=max(abs(a[k]-b[k]) for a,b in zip(aa,bb) for k in ['forget_loss','retain_loss','total_loss'])
 prior=base_samples();original_scored=json.loads((OUT/'original_development_protocol_check/samples.json').read_text());protocol={}
 for task,subjects in prior.items():
  refs={name:{r['doc_id']:r for r in records} for name,records in subjects.items()}
  diffs=[abs(r['scores'][k]-refs[r['task']][r['doc_id']]['resps'][k][0][0]) for r in original_scored[task] for k in range(4)]
  protocol[task]={'samples':len(original_scored[task]),'scored_choices':len(diffs),'max_loglikelihood_difference':max(diffs),'all_predictions_match':all(r['acc']==r['baseline_acc'] for r in original_scored[task])}
  assert max(diffs)==0
 dump(ROOT/'logs/rmu_adaptation_protocol_audit.json',protocol)
 audit={'baseline_protocol':protocol,'all_start_from_verified_original':True,'same_sample_exposure_sequences_across_runs':True,'same_normalized_control_direction_across_runs':True,'all_delta_save_reload_identical':True,'all_frozen_and_reference_parameters_unchanged':True,'current_control_vs_first_formal_first40_max_loss_difference':loss_diff,'run_count':len(runs),'development_checkpoint_count':len(rows)}
 dump(ROOT/f'logs/rmu_adaptation_audit{suffix}.json',audit)
 fields=['experiment','hook','update_parameters','learning_rate','alpha','steering_coefficient','step','scope','wmdp_bio_percent','mmlu_percent','wmdp_bio_change_pp','mmlu_change_pp','constraint_pass']
 table=[]
 for row in rows:
  table.append({**{k:row[k] for k in fields if k in row},'scope':'development:Bio256,MMLU1710','wmdp_bio_percent':row['wmdp_bio_development_percent'],'mmlu_percent':row['mmlu_development_percent'],'constraint_pass':row['development_constraint_pass']})
 for name,hook,step,metrics in [('original',None,0,{'wmdp_bio':{'accuracy_percent':73.84131971720346,'change_percentage_points':0},'mmlu':{'accuracy_percent':70.673693206096,'change_percentage_points':0}}),('first_formal80',7,80,{'wmdp_bio':{'accuracy_percent':59.937156323644935,'change_percentage_points':-13.904163393558523},'mmlu':{'accuracy_percent':65.89517162797323,'change_percentage_points':-4.778521578122774}})]:
  table.append(dict(experiment=name,hook=hook,update_parameters=[] if hook is None else ['layers5/6/7.mlp.down_proj.weight'],learning_rate=None if hook is None else 5e-5,alpha=None if hook is None else 1,steering_coefficient=None if hook is None else 2,step=step,scope='full',wmdp_bio_percent=metrics['wmdp_bio']['accuracy_percent'],mmlu_percent=metrics['mmlu']['accuracy_percent'],wmdp_bio_change_pp=metrics['wmdp_bio']['change_percentage_points'],mmlu_change_pp=metrics['mmlu']['change_percentage_points'],constraint_pass=metrics['mmlu']['change_percentage_points']>=-2))
 pick=result['selection']['selected'];full=result['full_metrics']
 if args.phase2:
  previous=json.loads((ROOT/'logs/rmu_adaptation_result.json').read_text());p1=previous['selection']['selected'];m1=previous['full_metrics']
  table.append({**{k:p1[k] for k in fields if k in p1},'scope':'phase1-full-exploratory','wmdp_bio_percent':m1['wmdp_bio']['accuracy_percent'],'mmlu_percent':m1['mmlu']['accuracy_percent'],'wmdp_bio_change_pp':m1['wmdp_bio']['change_percentage_points'],'mmlu_change_pp':m1['mmlu']['change_percentage_points'],'constraint_pass':previous['full_mmlu_constraint_pass']})
 if full:table.append({**{k:pick[k] for k in fields if k in pick},'scope':'full-exploratory:Bio1273,MMLU14042','wmdp_bio_percent':full['wmdp_bio']['accuracy_percent'],'mmlu_percent':full['mmlu']['accuracy_percent'],'wmdp_bio_change_pp':full['wmdp_bio']['change_percentage_points'],'mmlu_change_pp':full['mmlu']['change_percentage_points'],'constraint_pass':result['full_mmlu_constraint_pass']})
 dump(ROOT/f'logs/rmu_adaptation_comparison{suffix}.json',table)
 with (ROOT/f'logs/rmu_adaptation_comparison{suffix}.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(table)
 text='# Qwen3-4B RMU 适配实验结果\n\n'
 if args.phase2:text+='第一阶段全量候选MMLU下降2.3074点未达目标，之后按用户原授权分别追加lr2.5e-5与steering1，未同时修改两个因素。首轮全量参与追加动机及失败checkpoint排除，所有最终全量均为进一步探索。第一阶段完整结果保留于rmu-adaptation-results-phase1.md。\n\n'
 text+='各组均从同一原始模型独立开始，seed42、256token、batch1、累积16、40步；保存10/20/40步。详见 [审查与固定方案](rmu-adaptation-plan.md)。首轮80步作为历史结果列出，不能冒充本轮40步对照。\n\n'
 text+='| 实验 | hook | 更新层 down_proj.weight | lr | alpha | steering | 步数 | 范围 | Bio % | MMLU % | ΔBio pp | ΔMMLU pp | 2pp约束 |\n|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|\n'
 for t in table:
  layers='—' if t['hook'] is None else ','.join(map(str,range(t['hook']-2,t['hook']+1)))
  text+=f"| {t['experiment']} | {t['hook']} | {layers} | {t['learning_rate']} | {t['alpha']} | {t['steering_coefficient']} | {t['step']} | {t['scope']} | {t['wmdp_bio_percent']:.4f} | {t['mmlu_percent']:.4f} | {t['wmdp_bio_change_pp']:+.4f} | {t['mmlu_change_pp']:+.4f} | {'通过' if t['constraint_pass'] else '未通过'} |\n"
 text+='\n## 选择与解释\n\n'
 text+=f"开发集选择 `{pick['experiment']}` 第{pick['step']}步（hook{pick['hook']}、alpha{pick['alpha']}）：先筛MMLU相对子集基线下降不超过2个百分点，再选Bio最低；平手优先MMLU更高、步数更少。开发集原始模型Bio75.78125%，MMLU73.50877%，不能与全量原始分数混用。\n\n"
 if full:
  text+=f"全量探索性结果：Bio {full['wmdp_bio']['accuracy_percent']:.4f}%（变化{full['wmdp_bio']['change_percentage_points']:+.4f}个百分点），MMLU {full['mmlu']['accuracy_percent']:.4f}%（变化{full['mmlu']['change_percentage_points']:+.4f}个百分点）。本轮2个百分点筛选目标{'满足' if result['full_mmlu_constraint_pass'] else '未满足'}。{'在该有限预算和单seed范围内呈现更好的遗忘/保留平衡，但不能推广为普遍最优。' if result['full_mmlu_constraint_pass'] else '开发子集通过没有保证全量通过，不强行判为成功，按本阶段预先固定上限停止全量评测，不将未全量验证的其他开发候选当成已满足目标。'}\n\n"
 else:text+='所有候选在开发集均未满足约束，没有执行新的全量候选评测，不判为成功。\n\n'
 text+=f"审计：各组采样顺序一致、随机目标归一化方向一致，指定参数更新、冻结及参考参数不变、所有增量保存重载逐位一致。当前hook7对照与原正式运行前40步最大loss差为{loss_diff:g}。新最小检查脚本曾发生CPU/CUDA设备准备错误，修复后重试；失败日志独立保留，未修改原RMU训练行为。\n\n"
 text+='## 限制与下一步\n\n本轮没有独立WMDP-Bio验证题集：固定开发子集来自以前全量观察过的官方test，本轮全量结果只能称探索性。MMLU开发集每学科30题，子集均衡平均与全量样本加权分布不同，且存在抽样噪声。只用seed42，40步预算不代表80步或长期训练最优；首轮80步与选定40步全量比较同时涉及层位置和预算变化，不能把改善全部归因于换层；同seed、同采样不保证训练轨迹逐位复现，首轮与新对照的最大loss差已公开，不能将检查点保存/重载逐位一致混同为跨运行训练逐位一致；retain语料表示损失不等于独立正常能力验证。WMDP下降不证明知识已彻底删除，未做恢复攻击/再学习。\n\n下一步先保留本轮候选与失败结果，整理可加载交付元数据；若继续研究，先建立独立验证方案，再对选定少量配置做多seed及恢复测试。本轮最多7组40步、21个开发checkpoint、2个全量候选，没有开展大规模搜索；不能将本次官方test重新包装成未参与调参的独立测试。实际受限训练正文仍留本地，不公开上传。\n\n增量是BF16位级XOR+gzip自定义格式，需要固定原始基模及rmu_adaptation_utils.load_delta；不是LoRA。Git LFS工作树可能是pointer，真实文件仍在本地LFS cache和GitHub对象中；git lfs checkout可在空间允许时物化。优化器状态未保存，不宣称可从中间步继续训练。\n'
 (ROOT/'docs/rmu-adaptation-results.md').write_text(text)
 note='\n\n## RMU 小范围适配探索完成（2026-10-14，Asia/Shanghai；实际时间见run.json）\n\n'+text.split('## 选择与解释\n\n')[1].split('## 限制与下一步')[0]+'\n详细表见docs/rmu-adaptation-results.md，机器结果见logs/rmu_adaptation_result.json。固定开发子集来自官方test，全量为探索性，非独立测试。\n'
 # Use actual completion date from source metadata rather than hard-code schedule.
 date=max(x['completed_at'] for x in runs)[:10];note=note.replace('2026-10-14，Asia/Shanghai；实际时间见run.json',date+'，Asia/Shanghai')
 for name in ['01-WMDP-20261012-progress.md','prompt.md']:
  with (ROOT/name).open('a') as f:f.write(note)
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
