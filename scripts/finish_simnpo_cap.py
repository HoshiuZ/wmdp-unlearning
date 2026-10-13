"""Preserve old metrics; aggregate gradient-cap trial and prior gradient evidence."""
import json,csv,argparse,math,shutil,subprocess
from pathlib import Path
from rmu_adaptation_utils import ROOT,REPO,dump,materialized,digest
from reweight_rmu_development import summarize
OUT=REPO/'saves/unlearn/simnpo_rmu_v1';TRIAL=OUT/'joint_gradcap_screen40'
def read(p):return json.loads(p.read_text())
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--lock',action='store_true');parser.add_argument('--report',action='store_true');args=parser.parse_args();old=read(ROOT/'logs/simnpo_rmu_half_development.json');rows=old['rows'];baseline=read(ROOT/'logs/rmu_scale_v2_reweighted_development.json')['baseline'];new=[]
 for p in sorted(TRIAL.glob('checkpoint-*/development/samples.json')):
  data=read(p);m=summarize(data['mmlu']);r={'path':str(p.relative_to(ROOT)),'bio_percent':100*sum(x['acc'] for x in data['wmdp_bio'])/256,'mmlu':m,'equal_change_pp':m['all']['equal_percent']-baseline['all']['equal_percent'],'weighted_change_pp':m['all']['weighted_percent']-baseline['all']['weighted_percent']}
  for group in ['biology_related','other']:m[group]['weighted_change_pp']=m[group]['weighted_percent']-baseline[group]['weighted_percent']
  rows.append(r);new.append(r)
 for r in rows:r['pareto_weighted']=not any(q['bio_percent']<=r['bio_percent'] and q['mmlu']['all']['weighted_percent']>=r['mmlu']['all']['weighted_percent'] and (q['bio_percent']<r['bio_percent'] or q['mmlu']['all']['weighted_percent']>r['mmlu']['all']['weighted_percent']) for q in rows)
 best={str(t):min([r for r in rows if r['weighted_change_pp']>=-t],key=lambda r:(r['bio_percent'],-r['mmlu']['all']['weighted_percent'])) for t in [2,3]};dump(ROOT/'logs/simnpo_rmu_cap_development.json',{'rows':rows,'best_at_2_and_3pp':best,'exploratory':True,'weights':old['weights']})
 with (ROOT/'logs/simnpo_rmu_cap_development.csv').open('w') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['path','bio_percent','bio_change_pp','mmlu_equal','mmlu_weighted','weighted_change_pp','biology_change_pp','other_change_pp','pareto'])
  for r in rows:w.writerow([r['path'],r['bio_percent'],r['bio_percent']-75.78125,r['mmlu']['all']['equal_percent'],r['mmlu']['all']['weighted_percent'],r['weighted_change_pp'],r['mmlu']['biology_related']['weighted_change_pp'],r['mmlu']['other']['weighted_change_pp'],r['pareto_weighted']])
 print('CAP_DEV',[(r['path'].split('/')[-3],r['bio_percent'],r['weighted_change_pp'],r['pareto_weighted']) for r in new])
 if args.lock:
  assert len(new)==3 and read(TRIAL/'run.json')['status']=='completed';eligible=[r for r in new if r['pareto_weighted'] and r['weighted_change_pp']>=-3];selection=sorted(eligible,key=lambda r:(r['bio_percent'],-r['mmlu']['all']['weighted_percent']))[:1];p=ROOT/'logs/simnpo_rmu_cap_full_selection.json';assert not p.exists();dump(p,{'selected':selection,'rule':'New all-history development frontier with weighted MMLU loss<=2 or3; minBio,max1; fixed before full','exploratory':True});print('SELECTED',len(selection))
 if not args.report:return
 run=read(TRIAL/'run.json');oldrun=read(OUT/'joint_screen40/run.json');assert run['status']=='completed' and run['max_steps']==40 and run['gradient_cap'] and run['sim_scale']==1.;assert run['coefficients']['sim_weight']==oldrun['coefficients']['sim_weight'];assert run['coefficients']['ce_weight']==oldrun['coefficients']['ce_weight']
 for key in ['beta','delta','alpha_repr','steering','lr','seed','data_manifest']:assert run[key]==oldrun[key]
 for key in ['actual_input_tokens','actual_loss_tokens','effective_prediction_tokens']:assert run[key]==oldrun[key]
 assert read(TRIAL/'consumed_block_indices.json')==read(OUT/'joint_screen40/consumed_block_indices.json');assert read(TRIAL/'control_vector.json')==read(OUT/'joint_screen40/control_vector.json');assert run['frozen_parameters_unchanged'] and run['reference_unchanged'];assert read(OUT/'joint_gradcap_minimal/run.json')['status']=='completed'
 controller=read(TRIAL/'controller_metrics.json');assert len(controller)==640
 assert {s:sum(r['step']==s for r in controller) for s in range(40)}=={s:16 for s in range(40)}
 assert all(math.isfinite(r['effective_sim_weight']) and 0<=r['effective_sim_weight']<=run['coefficients']['sim_weight'] for r in controller)
 assert all(r['capped_ratio'] is None or r['capped_ratio']<=.500001 for r in controller)
 controller_stats={'n':len(controller),'cap_active_count':sum(r['cap_active'] for r in controller),'weight_min':min(r['effective_sim_weight'] for r in controller),'weight_max':max(r['effective_sim_weight'] for r in controller),'weight_mean':sum(r['effective_sim_weight'] for r in controller)/len(controller),'max_capped_ratio':max(r['capped_ratio'] for r in controller if r['capped_ratio'] is not None)}
 dump(ROOT/'logs/simnpo_rmu_cap_controller_summary.json',controller_stats)
 size=0
 for c in run['development']:
  assert c['save_reload_identical'];p=ROOT/c['checkpoint'];manifest=read(p/'checkpoint.json');assert len(manifest['parameters'])==3
  for r in manifest['parameters']:
   obj=materialized(p/r['file']);assert digest(obj.read_bytes())==r['compressed_sha256'] and obj.stat().st_size==r['bytes'];size+=r['bytes']
 reference=read(OUT.parent/'rmu_adaptation_v1/original_development_protocol_check/samples.json')
 for r in new:
  d=read(ROOT/r['path'])
  for task,n in [('wmdp_bio',256),('mmlu',1710)]:
   base={(x['task'],x['doc_id']):x for x in reference[task]};assert len(d[task])==n and {(x['task'],x['doc_id']) for x in d[task]}==set(base)
   for x in d[task]:
    y=base[(x['task'],x['doc_id'])];assert x['doc_hash']==y['doc_hash'] and x['prompt_hash']==y['prompt_hash'] and x['target']==y['target']
 diag=read(ROOT/'logs/simnpo_rmu_gradient_diagnostics.json');assert diag['no_optimizer_steps'] and diag['original_restored_fingerprints_equal'];selection=read(ROOT/'logs/simnpo_rmu_cap_full_selection.json')['selected'];full=[]
 for r in selection:
  cp=(ROOT/r['path']).parent.parent;full.append(read(OUT/'cap_full_evaluations'/(cp.parent.name+'_'+cp.name)/'result.json'))
 assert len(full)<=1;space=shutil.disk_usage(ROOT)._asdict();assert space['free']>2*2**30;dump(ROOT/'logs/simnpo_rmu_cap_completion_audit.json',{'single_factor_only':True,'controller_rule_verified':True,'controller_microbatches':640,'original_diagnostic_restored':True,'diagnostic_optimizer_steps':0,'new_training_cases':1,'steps':40,'checkpoints':3,'compressed_bytes':size,'matched_sampling_token_budget_control':True,'unchanged_frozen_reference':True,'all_reload_and_object_hashes_verified':True,'development_id_prompt_target_verified':True,'full_evaluations':len(full),'space':space,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
 lines=['# 训练中梯度尺度限幅对照结果','',
 '依据已完成的梯度诊断，执行1组训练中梯度尺度限幅对照。所有旧模型和结果保留；仅本地提交，未推送。','',
 '## 梯度证据','',
 '实际Qwen3-4B-Instruct-2507有36层；初版误写32层，已停止未更新权重的初次尝试，保留日志，正式诊断动态读取36层。诊断使用原始、RMU+CE40、联合20/40，各固定独立文章验证0/16/32/48；模型eval、padding/labels=-100排除，使用原保存控制向量及原固定权重。原始retain hook12输出no_grad缓存于RAM，不建立第二模型或保存表示。仅开启36层down_proj的requires_grad计算梯度，没有optimizer.step，末尾恢复并核对完整原始参数指纹。','',
 '| 状态 | 选定3层Sim/RMU加权梯度范数比均值 | RMU/Sim余弦均值 | 合成forget/retain余弦均值 |','| --- | ---: | ---: | ---: |']
 for name,records in diag['results'].items():
  gs=[r['groups']['selected'] for r in records];lines.append(f"| {name} | {sum(g['sim_over_rmu_weighted_norm'] for g in gs)/4:.4f} | {sum(g['cosines']['rmu_forget|sim_forget'] for g in gs)/4:.4f} | {sum(g['forget_retain_cosine'] for g in gs)/4:.4f} |")
 lines+=['','| 状态 | RMU加权梯度范数均值 | Sim加权梯度范数均值 | 表示保留加权范数均值 | CE保留加权范数均值 |','| --- | ---: | ---: | ---: | ---: |']
 for name,rs in diag['results'].items():
  means=[sum(r['groups']['selected']['weighted_norms'][k] for r in rs)/4 for k in ['rmu_forget','sim_forget','retain_repr','retain_ce']];lines.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in means)+' |')
 lines+=['','这里是每个probe比值的平均，不是先平均梯度范数再求比。两项遗忘梯度近正交、均值略正，未见稳定强反向冲突；forget与retain有部分负向冲突。原始接近零的微小负余弦不作强结论。明显证据是Sim加权梯度相对RMU在联合40步剧增，原始的一次尺度校准无法保证训练全过程平衡；并非所有损伤都能归因于方向冲突。','',
 '| 状态 | RMU梯度选定层能量占比 | Sim梯度选定层能量占比 | retain CE选定层能量占比 |','| --- | ---: | ---: | ---: |']
 for name,rs in diag['results'].items():
  frac=lambda k:100*sum(r['groups']['selected']['squared_norms'][k]/r['groups']['all']['squared_norms'][k] for r in rs)/4;lines.append(f"| {name} | {frac('rmu_forget'):.2f}% | {frac('sim_forget'):.2f}% | {frac('retain_ce'):.2f}% |")
 lines+=['','能量定义为梯度平方范数，仅比较同类型down_proj池，非全部模型参数。三层之外有大量梯度，但不代表这些方向有益、扩大容量可保持MMLU或有更好取舍。hook12表示项对13层及以后梯度为零的链路检查通过。本轮不扩大参数；四个文章probe有限，eval诊断不等于训练全部microbatch梯度，Adam/裁剪也未纳入一阶余弦。','',
 '## 单因素适配结果','',
 f"每个训练microbatch在实际可训练3个参数上测未加权RMU/Sim梯度范数，w=min({run['coefficients']['sim_weight']:.8f},0.5*R/(S+1e-12))，detach系数，不写入已有累计.grad。640条记录逐一验证限幅后比值<=0.5。限幅触发{controller_stats['cap_active_count']}/640，权重范围{controller_stats['weight_min']:.8f}–{controller_stats['weight_max']:.8f}，均值{controller_stats['weight_mean']:.8f}。",'',
 '保留项、原始起点、hook12、10/11/12 down_proj、alpha_repr2、steering2、beta0.1、delta0、CE系数、lr5e-5、seed42、40步、数据/采样均与固定原联合一致。验证使用最近训练权重，step0原始上界；验证/开发不更新控制器。只改权重计算规则，CUDA两步检查另从原始开始。动态loss标量随权重变化，不能直接与固定权重loss比较；未加权分项和controller逐批留档。单微批次优化器前约束不保证累积16后的范数比例、Adam更新或MMLU损失约束。','',
 '开发Bio256、MMLU每科30共1710，固定已观察官方test；基线Bio75.78125、等权MMLU73.50877、题数加权69.98410。加权使用57学科正式题数，不能消除小样本噪声；retain文章验证不是通用能力test。','',
 '| 步 | 原联合Bio% | 限幅Bio% | 原联合加权MMLUΔ点 | 限幅加权MMLUΔ点 | 限幅MMLU等权% | 总开发Pareto |','| --- | ---: | ---: | ---: | ---: | ---: | --- |']
 for r in new:
  step=(ROOT/r['path']).parent.parent.name;c=next(x for x in old['rows'] if '/joint_screen40/'+step+'/' in x['path']);lines.append(f"| {step} | {c['bio_percent']:.4f} | {r['bio_percent']:.4f} | {c['weighted_change_pp']:+.4f} | {r['weighted_change_pp']:+.4f} | {r['mmlu']['all']['equal_percent']:.4f} | {'是' if r['pareto_weighted'] else '否'} |")
 lines+=['','49个新/历史开发点全部保留并共同标记Pareto，见simnpo_rmu_cap_development.json/.csv；生物相关/其他学科和各科细项亦保留。','']
 for t,r in best.items():lines.append(f"损失≤{t}点的全体最佳开发点：`{r['path']}`，Bio{r['bio_percent']:.4f}%，加权变化{r['weighted_change_pp']:+.4f}点。")
 if not full:lines+=['','未出现同时满足预先前沿与≤2/3点损失规则的新增检查点，按规则不新增全量。不得把这些开发数字当全量独立test，也不把未入选解释为方法普遍无效。']
 else:lines+=['',f'锁定{len(full)}次全量探索性结果：`{json.dumps(full,ensure_ascii=False)}`；不依全量继续搜索。']
 v=run['validation'][-1];oldv=oldrun['validation'][-1]
 lines+=['','## 保存与限制','',f"最终retain文章NLL：限幅{v['retain_nll']:.4f}，原联合{oldv['retain_nll']:.4f}，原始2.1937。这是文章预测诊断，不能替代MMLU。",'',f"input/loss/有效预测token={run['actual_input_tokens']}/{run['actual_loss_tokens']}/{run['effective_prediction_tokens']}，与原联合精确匹配；每角色640次输入，采样顺序、控制向量一致。3个新增增量{size/2**20:.2f}MiB，重建逐位及真实LFS对象SHA通过；不保存optimizer，不能断点续训。运行{run['elapsed_seconds']:.1f}秒（含验证/开发回调），峰值allocated {run['cuda_peak_allocated_bytes']/2**30:.3f}GiB、reserved {run['cuda_peak_reserved_bytes']/2**30:.3f}GiB，训练代码{run['code_commit']}。诊断{diag['seconds']:.1f}秒、峰值{diag['gpu_peak_bytes']/2**30:.3f}GiB。完成时数据盘余{space['free']/2**30:.2f}GiB，无自动清理或扩容。",'',
 '完整配置、数据manifest、代码版本、实际采样、分项loss、验证与逐题评分均保存；原始及训练参数指纹、模型本地文件SHA固定，原始远端immutable revision仍未知。只做一组，没有追加rho或权重上界搜索、扩层、多seed或恢复测试；单seed、有限probe与已观察测试的探索不能证明知识删除。下一步根据实际比较决定是否值得研究动态权重或参数范围，不能把梯度占比当已验证方案。']
 (ROOT/'docs/simnpo-rmu-cap-results.md').write_text('\n'.join(lines)+'\n');print('REPORT_COMPLETE',size,space)
if __name__=='__main__':main()
