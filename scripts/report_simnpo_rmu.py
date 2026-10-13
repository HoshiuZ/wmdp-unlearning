"""Compile actual bounded ablation evidence; never launch new inference."""
import json,csv,shutil,subprocess
from pathlib import Path
from rmu_adaptation_utils import ROOT,REPO,dump,materialized,digest,base_samples
from reweight_rmu_development import summarize
OUT=REPO/'saves/unlearn/simnpo_rmu_v1'
def read(p):return json.loads(p.read_text())
def main():
 dev=read(ROOT/'logs/simnpo_rmu_development.json');lock=read(ROOT/'logs/simnpo_rmu_full_selection.json');coeff=read(ROOT/'logs/simnpo_rmu_coefficients.json');probe=read(OUT/'implementation_probe/run.json');assert probe['status']=='completed' and len(probe['gradient_checks'])==6
 new=[r for r in dev['rows'] if 'simnpo_rmu_v1' in r['path']];assert len(new)==9
 baseline_dev=read(REPO/'saves/unlearn/rmu_adaptation_v1/original_development_protocol_check/samples.json')
 for r in new:
  samples=read(ROOT/r['path'])
  for task,n in [('wmdp_bio',256),('mmlu',1710)]:
   reference={(x['task'],x['doc_id']):x for x in baseline_dev[task]};assert len(samples[task])==n
   assert {(x['task'],x['doc_id']) for x in samples[task]}==set(reference)
   for x in samples[task]:
    b=reference[(x['task'],x['doc_id'])];assert x['doc_hash']==b['doc_hash'] and x['prompt_hash']==b['prompt_hash'] and x['target']==b['target'] and x['baseline_acc']==b['acc']
 runs=[];size=0;count=0
 for mode in ['sim','rmu_ce','joint']:
  folder=OUT/(mode+'_screen40');run=read(folder/'run.json');assert run['status']=='completed';assert run['actual_input_tokens']==320619 and run['actual_loss_tokens']==319339 and run['effective_prediction_tokens']==319339;assert run['frozen_parameters_unchanged'] and run['reference_unchanged'];assert run['coefficients']==coeff['coefficients']
  consumed=read(folder/'consumed_block_indices.json');assert consumed==read(REPO/'saves/unlearn/rmu_adaptation_v1/h12_a1_lr5e-05_s2_screen40/consumed_block_indices.json');assert read(folder/'control_vector.json')['sha256']==read(REPO/'saves/unlearn/rmu_adaptation_v1/h12_a1_lr5e-05_s2_screen40/control_vector.json')['sha256']
  for row in run['development']:
   assert row['save_reload_identical'];cp=ROOT/row['checkpoint'];manifest=read(cp/'checkpoint.json');assert len(manifest['parameters'])==3
   for entry in manifest['parameters']:
    path=materialized(cp/entry['file']);assert path.stat().st_size==entry['bytes'] and digest(path.read_bytes())==entry['compressed_sha256'];size+=entry['bytes']
   count+=1
  component=read(folder/'component_metrics.json');assert len(component)==640;assert sum(r['effective_prediction_tokens'] for r in component)==319339;runs.append(run)
 assert count==9;full=[]
 for selected in lock['selected']:
  cp=(ROOT/selected['path']).parent.parent;folder=OUT/'full_evaluations'/(cp.parent.name+'_'+cp.name);result=read(folder/'result.json');samples=read(folder/'samples.json')
  for task,n in [('wmdp_bio',1273),('mmlu',14042)]:
   assert len(samples[task])==n and len({(r['task'],r['doc_id']) for r in samples[task]})==n;assert sum(r['acc'] for r in samples[task])==result['metrics'][task]['correct']
  base=base_samples()
  for task in samples:
   reference={(name,r['doc_id']):r for name,rs in base[task].items() for r in rs}
   for r in samples[task]:
    b=reference[(r['task'],r['doc_id'])];assert r['doc_hash']==b['doc_hash'] and r['prompt_hash']==b['prompt_hash'] and r['baseline_acc']==b['acc']
  full.append({'name':folder.name,**result,'mmlu_subjects':summarize(samples['mmlu'])})
 assert len(full)<=2;space=shutil.disk_usage(ROOT)._asdict();assert space['free']>2*2**30
 audit={'new_training_cases':3,'steps_per_case':40,'saved_checkpoints':count,'compressed_delta_bytes':size,'full_evaluations':len(full),'all_reload_identical':True,'all_object_sha_verified':True,'frozen_and_reference_unchanged':True,'matched_sampling_and_input_loss_prediction_token_budget':True,'all_development_question_prompt_and_target_identity_verified':True,'space':space,'optimizer_saved':False,'independent_test':False,'full_sample_hashes_verified':bool(full),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()};dump(ROOT/'logs/simnpo_rmu_completion_audit.json',audit);dump(ROOT/'logs/simnpo_rmu_full_results.json',{'rows':full,'exploratory':True})
 lines=['# SimNPO独立对照与RMU联合：有限探索结果','',
 '三组均从同一Qwen3-4B-Instruct-2507原始模型开始；hook12、仅更新10/11/12 down_proj（74,711,040参数）、lr5e-5、seed42、batch1×累积16、256token、40步、同formal_blocks_v1训练文章/块/顺序。新增B=SimNPO+retain CE；C=RMU+retain表示MSE+retain CE；D=与C相同再加SimNPO遗忘项。历史A为alpha2/steering2 RMU，原文件保留。','',
 '## 实现与预先锁定尺度','',
 '源框架SimNPO已修复非模型index过滤及按labels[:,1:]计算有效token分母，零有效token报错。本轮块首token本来就是-100，所以分母修复不改变这些块的计数。自定义ControlledSimRMU共享forget/retain前向，SimNPO和retain CE在FP32 logits上计算，平均方式为每篇预测token平均后batch平均；与旧RMU的表示MSE不混作同一个loss标量。所有预测loss排除padding/labels=-100。最小CUDA各方法两步证明指定参数梯度有限非零、仅指定参数改变、参考模型冻结eval/no_grad、其余参数不变；mask和移位的单元检查也通过。','',
 f"固定beta=0.1、delta=0。原始模型独立文章验证0/16/32/48的梯度校准得到SimNPO系数{coeff['coefficients']['sim_weight']:.8f}、retain CE系数{coeff['coefficients']['ce_weight']:.8f}；目标分别是原始RMU forget梯度均值的0.5倍/0.25倍。所有新组使用同系数，C/D还用alpha_repr=2、steering2及原保存控制方向；B不使用RMU目标或参考表示。校准不是通过评测选择超参，仅是有限probe上的尺度对齐，不保证训练过程相同梯度比例。",'',
 'B为参数范围受控的SimNPO变体，而非全参数SimNPO公开榜单复现。为了沿用同一个审计/冻结基础类，B仍持有未参与loss的冻结参考模型，并计算未加入loss的RMU表示诊断，因此实际资源用量不是原生SimNPO的最低需求。没有拿这一实现的时间/显存宣称SimNPO固有更快。C/D使用相同保留项，D新增SimNPO也改变总梯度，收益不能排除总体遗忘强度因素。联合实现为本项目探索，不是框架现成方法或已证明更优算法。','',
 '## 开发评测','',
 '固定Bio256、MMLU57科×30=1710；MMLU同时报等权及按全量科目题数加权，同原模型权重，生物相关10科定义与上一轮相同。原开发基线Bio75.78125%、MMLU等权73.50877%、加权69.98410%；小样本加权不能消除抽样误差。文章holdout64/role用于损失/NLL，不等于独立能力test；官方测试已被观察，所有指标探索性。','',
 '| 新组/步 | Bio% | MMLU等权% | MMLU加权% | 加权Δ点 | 生物相关Δ点 | 其他Δ点 | 总开发Pareto |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
 for r in new:
  p=ROOT/r['path'];lines.append(f"| {p.parent.parent.parent.name}/{p.parent.parent.name} | {r['bio_percent']:.4f} | {r['mmlu']['all']['equal_percent']:.4f} | {r['mmlu']['all']['weighted_percent']:.4f} | {r['weighted_change_pp']:+.4f} | {r['mmlu']['biology_related']['weighted_change_pp']:+.4f} | {r['mmlu']['other']['weighted_change_pp']:+.4f} | {'是' if r['pareto_weighted'] else '否'} |")
 lines+=['','总Pareto比较涵盖所有历史开发点与新点；完整43点及各科细项见simnpo_rmu_development.json/.csv，没有覆盖上一轮报告。','']
 for t,r in dev['best_all_at_2_and_3pp'].items():lines.append(f"- 全体已观测开发点、MMLU加权损失≤{t}点的最低Bio：`{r['path']}`，Bio{r['bio_percent']:.4f}%，加权变化{r['weighted_change_pp']:+.4f}点。")
 # Direct C/D at equal step, independent of old-frontier selection.
 lines+=['','| 同步数D减C | BioΔ点 | 加权MMLUΔ点 |','| --- | ---: | ---: |']
 for step in [10,20,40]:
  c=next(r for r in new if f'/rmu_ce_screen40/checkpoint-{step}/' in r['path']);d=next(r for r in new if f'/joint_screen40/checkpoint-{step}/' in r['path']);lines.append(f"| {step} | {d['bio_percent']-c['bio_percent']:+.4f} | {d['mmlu']['all']['weighted_percent']-c['mmlu']['all']['weighted_percent']:+.4f} |")
 lines+=['','D减C的负Bio/正MMLU才表示在此开发样本上双指标改进；一个好一个坏是取舍，不能只凭Bio称组合成功。A与C保留目标不同，仅作保留项消融。','',
 '## 全量评测与筛选','',f"依据预先规则锁定{len(lock['selected'])}个新增非支配且加权损失≤2/3点的候选，去重后最多2次。没有根据全量结果追加搜索，也没有额外纳入超过3点模型。",'']
 if full:
  lines+=['| 新全量候选 | Bio% | BioΔ点 | MMLU% | MMLUΔ点 | ≤2点 | ≤3点 |','| --- | ---: | ---: | ---: | ---: | --- | --- |']
  for r in full:
   b=r['metrics']['wmdp_bio'];m=r['metrics']['mmlu'];lines.append(f"| {r['name']} | {b['accuracy_percent']:.4f} | {b['change_percentage_points']:+.4f} | {m['accuracy_percent']:.4f} | {m['change_percentage_points']:+.4f} | {'是' if r['loss_le_2pp'] else '否'} | {'是' if r['loss_le_3pp'] else '否'} |")
 else:lines+=['新增点未满足预先声明的全量入选条件，因此本轮没有新增全量评测。这不证明这些方法绝对无效，只表示该参数范围、单seed和40步预算下未产生值得新增全量的开发前沿点。']
 lines+=['','历史全量仍保留：原始Bio73.8413/MMLU70.6737；RMU alpha2/steering2 Bio66.3001/MMLU69.1355，损失1.5382点；RMU alpha2/steering32 Bio55.3810/MMLU68.4874，损失2.1863点。开发不能与这些全量数字直接混用。所有新增全量也不代表独立测试，不证明知识删除。','',
 '## 时间、资源与保存','',
 '本轮三个新增目标共120正式更新步，CUDA实现检查另有各2步并从原始权重独立恢复，不把它们接入正式训练。依赖沿用PyTorch2.9.1+cu128、Transformers5.5.4、lm_eval0.4.11、datasets3.0.1、Accelerate1.13.0。执行版本见各run.json和simnpo_rmu_provenance.json；probe过程的token计数遥测修订不改loss/梯度。B路径配置中未使用的历史paths.output_dir保留原样，实际Trainer output_dir与真实保存目录一致。','',
 '| 新组 | 代码提交 | 运行秒数 | 峰值allocated GiB | reserved GiB | input/loss/预测token |','| --- | --- | ---: | ---: | ---: | --- |']
 for r in runs:lines.append(f"| {r['variant']} | {r['code_commit'][:7]} | {r['elapsed_seconds']:.1f} | {r['cuda_peak_allocated_bytes']/2**30:.3f} | {r['cuda_peak_reserved_bytes']/2**30:.3f} | {r['actual_input_tokens']}/{r['actual_loss_tokens']}/{r['effective_prediction_tokens']} |")
 lines+=['','运行时间和Trainer train_runtime包含验证/开发回调，非纯前后向时间。每组10/20/40无损增量，共9个，重载逐位一致；3组每角色640次输入，同历史序列及实际预算。完整配置、数据SHA、分项微批loss640条/组、每10步64块/role文章验证、token数、显存、指纹和逐题评分均保存。未保存optimizer，不可断点续训；没有安装无关依赖或保存完整模型副本。','',f"新增实际压缩权重{size/2**20:.2f}MiB；完成时数据盘余{space['free']/2**30:.2f}GiB，未删除其他模型或自动扩容。基模远端immutable revision未知，以原本地文件清单及完整参数指纹固定。代码、测试、方案和日志仅本地提交，推送按用户指令。",'',
 '## 限制与下一步','',
 '第40步联合组比同保留项RMU组进一步降低Bio6.2500点，却额外损失加权MMLU1.9643点；retain文章验证NLL升到3.0356（原始2.1937），而对照为2.1877。独立SimNPO的验证NLL为2.6023。本设置下联合更强地削弱了Bio，但能力损伤更大；不建议替换现有已验证RMU配置。全部新增开发点均被既有候选支配，未显示新的取舍前沿。','',
 '这是固定三层、单seed、一个beta和梯度校准系数的有限探索；不能外推方法的普遍优劣。联合有合理的表示/输出互补假说，但结果决定是否值得继续，不预设成功。不自动增加学习率、参数范围或超参搜索，也不自动多seed/恢复测试。若有新的良好取舍，下一轮可预先确定少量复验及恢复评测；若没有，应先检查输出梯度、同域forget/retain冲突和受限参数容量，再单独改变一个因素。','',
 '方法来源：[SimNPO官方仓库](https://github.com/OPTML-Group/Unlearn-Simple)，[论文](https://arxiv.org/abs/2410.07163)；RMU及OpenUnlearning来源见根README。']
 (ROOT/'docs/simnpo-rmu-results.md').write_text('\n'.join(lines)+'\n');print('AUDIT',audit);print('FULL',full)
if __name__=='__main__':main()
