"""Aggregate the predeclared steering32 matched pair without changing old results."""
import json,csv,argparse,math,shutil
from rmu_adaptation_utils import ROOT,REPO,dump,materialized,digest,base_samples
from reweight_rmu_development import summarize
OUT=REPO/'saves/unlearn/simnpo_rmu_v1'
NAMES=['rmu_ce_s32_screen40','joint_gradcap_s32_screen40']
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--lock',action='store_true');p.add_argument('--report',action='store_true');a=p.parse_args()
 old=read(ROOT/'logs/simnpo_rmu_cap_development.json');rows=old['rows'];baseline=read(ROOT/'logs/rmu_scale_v2_reweighted_development.json')['baseline'];new=[];size=0;runs=[]
 ref=read(OUT.parent/'rmu_adaptation_v1/original_development_protocol_check/samples.json')
 for name in NAMES:
  d=OUT/name;run=read(d/'run.json');runs.append(run);assert run['status']=='completed' and run['steering']==32 and run['max_steps']==40
  assert read(OUT/name.replace('_screen40','_minimal')/'run.json')['status']=='completed'
  legacy=read(OUT/'rmu_ce_screen40/run.json')
  for k in ['beta','delta','alpha_repr','lr','seed','coefficients','actual_input_tokens','actual_loss_tokens','effective_prediction_tokens']:assert run[k]==legacy[k]
  assert run['frozen_parameters_unchanged'] and run['reference_unchanged']
  assert read(d/'consumed_block_indices.json')==read(OUT/'rmu_ce_screen40/consumed_block_indices.json')
  cv=read(d/'control_vector.json');orig=read(OUT/'rmu_ce_screen40/control_vector.json');assert cv['values']==[x*16 for x in orig['values']] and abs(cv['l2_norm']-32)<1e-4
  for c in run['development']:
   assert c['save_reload_identical'];cp=ROOT/c['checkpoint'];manifest=read(cp/'checkpoint.json');assert len(manifest['parameters'])==3
   for q in manifest['parameters']:
    obj=materialized(cp/q['file']);assert digest(obj.read_bytes())==q['compressed_sha256'] and obj.stat().st_size==q['bytes'];size+=q['bytes']
  for path in sorted(d.glob('checkpoint-*/development/samples.json')):
   data=read(path)
   for task,n in [('wmdp_bio',256),('mmlu',1710)]:
    base={(x['task'],x['doc_id']):x for x in ref[task]};assert len(data[task])==n and {(x['task'],x['doc_id']) for x in data[task]}==set(base)
    for x in data[task]:
     y=base[x['task'],x['doc_id']];assert all(x[k]==y[k] for k in ['doc_hash','prompt_hash','target'])
   m=summarize(data['mmlu']);r={'path':str(path.relative_to(ROOT)),'bio_percent':100*sum(x['acc'] for x in data['wmdp_bio'])/256,'mmlu':m,'equal_change_pp':m['all']['equal_percent']-baseline['all']['equal_percent'],'weighted_change_pp':m['all']['weighted_percent']-baseline['all']['weighted_percent']}
   for group in ['biology_related','other']:m[group]['weighted_change_pp']=m[group]['weighted_percent']-baseline[group]['weighted_percent']
   rows.append(r);new.append(r)
 assert len(new)==6 and len(rows)==55
 assert read(OUT/NAMES[0]/'control_vector.json')==read(OUT/NAMES[1]/'control_vector.json')
 controller=read(OUT/NAMES[1]/'controller_metrics.json');assert len(controller)==640
 assert all(sum(r['step']==s for r in controller)==16 for s in range(40))
 assert all(math.isfinite(r['effective_sim_weight']) and 0<=r['effective_sim_weight']<=runs[1]['coefficients']['sim_weight'] and (r['capped_ratio'] is None or r['capped_ratio']<=.500001) for r in controller)
 for r in rows:r['pareto_weighted']=not any(q['bio_percent']<=r['bio_percent'] and q['mmlu']['all']['weighted_percent']>=r['mmlu']['all']['weighted_percent'] and (q['bio_percent']<r['bio_percent'] or q['mmlu']['all']['weighted_percent']>r['mmlu']['all']['weighted_percent']) for q in rows)
 dump(ROOT/'logs/simnpo_rmu_s32_development.json',{'rows':rows,'weights':old['weights'],'exploratory':True})
 with (ROOT/'logs/simnpo_rmu_s32_development.csv').open('w') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['path','hook','parameters','lr','alpha','steering','steps','bio','mmlu_weighted','mmlu_equal','weighted_change_pp','pareto'])
  for r in new:w.writerow([r['path'],12,'10/11/12 down_proj.weight',5e-5,2,32,int(r['path'].split('checkpoint-')[1].split('/')[0]),r['bio_percent'],r['mmlu']['all']['weighted_percent'],r['mmlu']['all']['equal_percent'],r['weighted_change_pp'],r['pareto_weighted']])
 print('NEW',[(r['path'].split('/')[-4:-2],r['bio_percent'],r['weighted_change_pp'],r['pareto_weighted']) for r in new],flush=True)
 selection_path=ROOT/'logs/simnpo_rmu_s32_full_selection.json'
 if a.lock:
  assert not selection_path.exists();selected=[]
  for t in [2,3]:
   candidates=[r for r in new if r['pareto_weighted'] and r['weighted_change_pp']>=-t]
   if candidates:
    best=min(candidates,key=lambda r:(r['bio_percent'],-r['mmlu']['all']['weighted_percent']))
    if not any(r['path']==best['path'] for r in selected):selected.append(best)
  dump(selection_path,{'selected':selected,'rule':'Predeclared new global Pareto, minimum Bio at <=2pp and <=3pp loss, deduplicate max2; exploratory','exploratory':True});print('SELECTED',len(selected),flush=True)
 if not a.report:return
 selected=read(selection_path)['selected'];full=[]
 for r in selected:
  cp=(ROOT/r['path']).parent.parent;dest=OUT/'s32_full_evaluations'/(cp.parent.name+'_'+cp.name);full.append(read(dest/'result.json'))
  data=read(dest/'samples.json')
  for task,subjects in base_samples().items():
   base={(name,x['doc_id']):x for name,records in subjects.items() for x in records};assert len(data[task])==len(base) and {(x['task'],x['doc_id']) for x in data[task]}==set(base)
   for x in data[task]:
    y=base[x['task'],x['doc_id']];assert x['doc_hash']==y['doc_hash'] and x['prompt_hash']==y['prompt_hash'] and x['target']==int(y['doc']['answer']) and x['baseline_acc']==int(y['acc'])

 audit={'cases':NAMES,'checkpoints':6,'compressed_bytes':size,'reference_frozen_unchanged':True,'same_sampling_tokens_control':True,'reload_object_sha_verified':True,'development_requests_verified':True,'full_requests_verified':True,'controller_records':640,'cap_active':sum(r['cap_active'] for r in controller),'max_capped_ratio':max(r['capped_ratio'] for r in controller if r['capped_ratio'] is not None),'full_results':full,'space':shutil.disk_usage(ROOT)._asdict()};dump(ROOT/'logs/simnpo_rmu_s32_completion_audit.json',audit)
 lines=['# steering32匹配对照结果','','两组均从原始模型独立训练40步；hook12，10/11/12 down_proj.weight，lr5e-5，alpha2，steering32，seed42，256token，batch1/累积16。CE固定0.0600289086；联合Sim上界0.1566522562、beta0.1、delta0、rho0.5。组间只增加限幅Sim项。','', '| 开发实验 | 步数 | Bio% | MMLU题数加权% | MMLU等权% | 加权MMLU变化点 | 历史共同前沿 |','| --- | ---: | ---: | ---: | ---: | ---: | --- |']
 for r in new:
  cp=(ROOT/r['path']).parent.parent;lines.append(f"| {cp.parent.name} | {cp.name} | {r['bio_percent']:.4f} | {r['mmlu']['all']['weighted_percent']:.4f} | {r['mmlu']['all']['equal_percent']:.4f} | {r['weighted_change_pp']:+.4f} | {'是' if r['pareto_weighted'] else '否'} |")
 lines+=['','同预算匹配比较（联合减RMU+CE）：','', '| 步 | Bio变化点 | 加权MMLU变化点 |','| --- | ---: | ---: |']
 for step in [10,20,40]:
  c=next(r for r in new if '/rmu_ce_s32_screen40/checkpoint-'+str(step)+'/' in r['path']);j=next(r for r in new if '/joint_gradcap_s32_screen40/checkpoint-'+str(step)+'/' in r['path']);lines.append(f"| {step} | {j['bio_percent']-c['bio_percent']:+.4f} | {j['weighted_change_pp']-c['weighted_change_pp']:+.4f} |")
 lines+=['','单因素steering2→32（其余条件一致）：','', '| 实验 | 步 | s2 Bio% | s32 Bio% | s2 MMLU变化点 | s32 MMLU变化点 |','| --- | ---: | ---: | ---: | ---: | ---: |']
 for name,previous in zip(NAMES,['rmu_ce_screen40','joint_gradcap_screen40']):
  for step in [10,20,40]:
   r=next(r for r in new if '/'+name+'/checkpoint-'+str(step)+'/' in r['path']);q=next(r for r in old['rows'] if '/'+previous+'/checkpoint-'+str(step)+'/' in r['path']);lines.append(f"| {name} | {step} | {q['bio_percent']:.4f} | {r['bio_percent']:.4f} | {q['weighted_change_pp']:+.4f} | {r['weighted_change_pp']:+.4f} |")
 lines+=['','开发集Bio256、MMLU1710，基线75.78125%/69.98410%（加权）。保存全部55个历史/新增点及生物相关、其他和每科学科结果；开发来自已观察官方test，retain文章验证不是MMLU独立验证。','']
 if not full:lines+=['新点未通过预先声明的总体前沿与能力约束筛选，不追加全量评测；不能以开发准确率声称改善最终全量结果。']
 else:
  lines+=['全量结果仍为探索性（官方test已参与调参）：','', '| 检查点 | Bio% | MMLU% | MMLU变化点 | <=2点 |','| --- | ---: | ---: | ---: | --- |']
  for f in full:
   m=f['metrics'];lines.append(f"| {f['checkpoint']} | {m['wmdp_bio']['accuracy_percent']:.4f} | {m['mmlu']['accuracy_percent']:.4f} | {m['mmlu']['change_percentage_points']:+.4f} | {f['loss_le_2pp']} |")
 if full:
  passing=[f for f in full if f['loss_le_2pp']]
  if passing:
   winner=min(passing,key=lambda f:f['metrics']['wmdp_bio']['accuracy_percent']);m=winner['metrics'];lines+=['',f"本轮选择RMU+CE第40步作为严格2点档最佳已观察候选：Bio{m['wmdp_bio']['accuracy_percent']:.4f}%、MMLU{m['mmlu']['accuracy_percent']:.4f}%。相比此前合格RMU alpha2/s2，Bio再低{66.30007855459544-m['wmdp_bio']['accuracy_percent']:.4f}点，MMLU低{69.13545079048569-m['mmlu']['accuracy_percent']:.4f}点；距离2点上限仅余{2+m['mmlu']['change_percentage_points']:.4f}点，不能声称多seed稳定满足。",'', '联合40步虽在开发集多降低Bio3.5156点，却多损失加权MMLU2.2219点，未达到2点或3点筛选档，因此未全量评测。新增Sim项没有展示更好的匹配取舍；新候选仍属于RMU加保留正则，不应宣称Sim联合已成功。下一步建议先对新RMU+CE候选做另一个固定seed复验，并先确定独立筛选/验证方案；本轮不自动执行。']
 lines+=['','历史全量参考：原始Bio73.8413%、MMLU70.6737%；RMU alpha2/s32无额外CE为55.3810%/68.4874%，MMLU下降2.1863点；严格2点约束最佳已观测RMU alpha2/s2为66.3001%/69.1355%。不能把额外CE收益归因于Sim。','',f"限幅触发{audit['cap_active']}/640，最大微批次比例{audit['max_capped_ratio']:.6f}。只约束单微批次原始梯度，不保证Adam、累积或MMLU约束；未加权分项与实际权重逐批保留。两组input/loss/prediction=320619/319339/319339，采样、方向、预算一致；冻结参数/参考不变，6增量逐位重建及真实LFS对象SHA通过，合计{size/2**20:.2f}MiB。没有optimizer状态，不能断点续训。",'']
 for run in runs:lines.append(f"{run['name']}：{run['elapsed_seconds']:.1f}秒（含回调），峰值allocated {run['cuda_peak_allocated_bytes']/2**30:.3f}GiB/reserved {run['cuda_peak_reserved_bytes']/2**30:.3f}GiB；代码{run['code_commit']}；retain NLL {run['validation'][-1]['retain_nll']:.4f}。")
 lines+=['','完整配置、数据manifest、代码版本、控制向量、逐题请求、损失及指纹均留档。固定原始模型本地SHA；远端不可变revision仍未知。单seed、小开发集、已观察测试存在限制，WMDP下降不证明知识彻底删除。未追加系数/层/多seed搜索，未自动推送。下一步应根据匹配结果判断是否保留Sim联合路线；如继续验证，先确定独立验证方案与计算预算。']
 (ROOT/'docs/simnpo-rmu-s32-results.md').write_text('\n'.join(lines)+'\n');print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
