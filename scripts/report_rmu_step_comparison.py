"""Recompute fixed step comparison from public per-question records, no inference."""
import csv,hashlib,json,shutil,subprocess
from pathlib import Path
from rmu_adaptation_utils import ROOT,REPO,dump,base_samples
from rmu_step_common import RUN,HISTORY,STEPS,SELECTED
from reweight_rmu_development import summarize,BIO_SUBJECTS

def flat(data):return {t:rs if isinstance(rs,list) else [dict(r,task=s) for s,rows in rs.items() for r in rows] for t,rs in data.items()}
def aggregate(name,step,data,baseline):
 data=flat(data);m=summarize(data['mmlu']);base=flat(baseline);bm=summarize(base['mmlu']);bio=100*sum(r['acc'] for r in data['wmdp_bio'])/len(data['wmdp_bio']);bbio=100*sum(r['acc'] for r in base['wmdp_bio'])/len(base['wmdp_bio']);mm=100*sum(r['acc'] for r in data['mmlu'])/len(data['mmlu']);bmm=100*sum(r['acc'] for r in base['mmlu'])/len(base['mmlu'])
 return {'experiment':name,'step':step,'bio_percent':bio,'bio_drop_pp':bbio-bio,'mmlu_percent':mm,'mmlu_loss_pp':bmm-mm,'mmlu_equal_percent':m['all']['equal_percent'],'mmlu_weighted_percent':m['all']['weighted_percent'],'mmlu_equal_change_pp':m['all']['equal_percent']-bm['all']['equal_percent'],'mmlu_weighted_change_pp':m['all']['weighted_percent']-bm['all']['weighted_percent'],'biology_related':dict(m['biology_related'],change_pp=m['biology_related']['weighted_percent']-bm['biology_related']['weighted_percent']),'other':dict(m['other'],change_pp=m['other']['weighted_percent']-bm['other']['weighted_percent']),'subjects':{s:dict(v,change_pp=v['accuracy_percent']-bm['per_subject'][s]['accuracy_percent']) for s,v in m['per_subject'].items()}}

def main():
 out=RUN/'full_evaluations';train=json.loads((RUN/'run.json').read_text());ev=json.loads((out/'evaluation_run.json').read_text());assert train['status']=='completed' and 'completed_at' in ev
 baseline=base_samples();original=aggregate('original',0,baseline,baseline)
 historical=aggregate('historical40',40,json.loads((REPO/'saves/unlearn/rmu_scale_v2/full_evaluations/h12_a2_s32_fixed40_checkpoint-40/samples.json').read_text()),baseline)
 rows=[aggregate(f'continuous_step{step}',step,json.loads((out/f'checkpoint-{step}/samples.json').read_text()),baseline) for step in STEPS]
 for r in rows:r['dominated_by']=[q['step'] for q in rows if q['bio_percent']<=r['bio_percent'] and q['mmlu_percent']>=r['mmlu_percent'] and (q['bio_percent']<r['bio_percent'] or q['mmlu_percent']>r['mmlu_percent'])]
 feasible={str(t):[r['step'] for r in rows if r['mmlu_loss_pp']<=t+1e-10] for t in [2,3]};pareto=[r['step'] for r in rows if not r['dominated_by']]
 changes=[{'from':a['step'],'to':b['step'],'bio_change_pp':b['bio_percent']-a['bio_percent'],'mmlu_change_pp':b['mmlu_percent']-a['mmlu_percent'],'biology_change_pp':b['biology_related']['weighted_percent']-a['biology_related']['weighted_percent'],'other_change_pp':b['other']['weighted_percent']-a['other']['weighted_percent']} for a,b in [(rows[0],rows[1]),(rows[1],rows[2]),(rows[0],rows[2])]]
 devbase=json.loads((REPO/'saves/unlearn/rmu_adaptation_v1/original_development_protocol_check/samples.json').read_text());dev=[aggregate(f'continuous_step{step}',step,json.loads((RUN/f'checkpoint-{step}/development/samples.json').read_text()),devbase) for step in STEPS]
 devbaseline=aggregate('development_original',0,devbase,devbase)
 local=[]
 for step in STEPS:
  checkpoint=RUN/f'checkpoint-{step}';manifest=json.loads((checkpoint/'checkpoint.json').read_text())
  for p in manifest['parameters']:
   oid=p['compressed_sha256'];cache=ROOT/'.git/lfs/objects'/oid[:2]/oid[2:4]/oid;assert cache.is_file() and cache.stat().st_size==p['bytes']
   local.append({'step':step,'pointer_path':str((checkpoint/p['file']).relative_to(ROOT)),'local_object_path':str(cache),'oid_sha256':oid,'bytes':p['bytes'],'remote_uploaded_this_round':False,'remote_verified':False})
 dump(RUN/'delta_storage_status.json',{'quota':'unknown','reason':'Git SSH transport available; no authenticated billing/LFS quota interface available in this environment. Existing uploads and disk free space do not prove remaining quota.','upload_policy':'Push text only. New delta pointers are locally retained and ignored by Git until quota is confirmed.','objects':local,'total_payload_bytes':sum(x['bytes'] for x in local),'optimizer_state_saved':False})
 report_pause=json.loads((RUN/'reproduction40_decision.json').read_text()).get('audit_pause_seconds_before_continue',0) if (RUN/'reproduction40_decision.json').exists() else 0
 before=train['space_before'];disk=shutil.disk_usage(ROOT);allocated=sum(p.stat().st_blocks*512 for p in RUN.rglob('*') if p.is_file());lfs_alloc=sum(Path(p['local_object_path']).stat().st_blocks*512 for p in local)
 resource={'total_training_process_seconds':train['total_process_seconds'],'trainer_wall_seconds_including_callbacks':train['trainer_train_wall_seconds_including_callbacks'],'optimizer_update_loop_seconds':train['optimizer_update_loop_seconds'],'callback_seconds_including_checks':sum(r['seconds_including_integrity_checks'] for r in train['callback_integrity']),'new_full_inference_seconds':sum(json.loads((out/f'checkpoint-{step}/summary.json').read_text())['seconds'] for step in STEPS),'reproduction_audit_pause_seconds':report_pause,'evaluation_wall_seconds_including_loading_and_checks':ev['total_wall_seconds_including_loading_and_checks'],'training_peak_allocated_bytes':train['cuda_peak_allocated_bytes'],'training_peak_reserved_bytes':train['cuda_peak_reserved_bytes'],'evaluation_peak_allocated_bytes':ev['cuda_peak_allocated_bytes'],'new_run_allocated_bytes':allocated,'new_delta_allocated_bytes':lfs_alloc,'new_delta_payload_bytes':sum(x['bytes'] for x in local),'free_bytes':disk.free,'free_space_change_since_run_start_bytes':before['free']-disk.free}
 report={'original':original,'historical40':historical,'trajectory':rows,'development_baseline':devbaseline,'development':dev,'changes':changes,'feasible_steps_by_mmlu_loss_pp':feasible,'pareto_steps':pareto,'bio_lowest_steps':[r['step'] for r in rows if r['bio_percent']==min(q['bio_percent'] for q in rows)],'resources':resource,'reproduction40':json.loads((RUN/'reproduction40.json').read_text()),'biology_subject_definition':sorted(BIO_SUBJECTS),'limitations':['Official test has participated in prior exploration.','Single seed exploratory results.','WMDP accuracy reduction is not proof of complete knowledge deletion.','Delta weights do not include optimizer state and cannot independently resume the exact continuous trajectory.','New deltas are local only: remote quota unknown, no upload or remote backup claimed.']}
 dump(ROOT/'logs/rmu_step_comparison_results.json',report)
 fields=['experiment','step','bio_percent','bio_drop_pp','mmlu_percent','mmlu_loss_pp']
 with (ROOT/'logs/rmu_step_comparison_full.csv').open('w') as f:
  w=csv.DictWriter(f,fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows([original,historical,*rows])
 with (ROOT/'logs/rmu_step_comparison_development.csv').open('w') as f:
  keys=['experiment','step','bio_percent','bio_drop_pp','mmlu_equal_percent','mmlu_weighted_percent','mmlu_equal_change_pp','mmlu_weighted_change_pp'];w=csv.DictWriter(f,keys,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows([devbaseline,*dev])
 with (ROOT/'logs/rmu_step_comparison_subjects.csv').open('w') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['step','subject','group','n','accuracy_percent','change_from_base_pp','change_from_step40_pp','change_from_step80_pp'])
  for r in rows:
   for s,v in sorted(r['subjects'].items()):w.writerow([r['step'],s,'biology10' if s in BIO_SUBJECTS else 'other47',v['dev_n'],v['accuracy_percent'],v['change_pp'],v['accuracy_percent']-rows[0]['subjects'][s]['accuracy_percent'],v['accuracy_percent']-rows[1]['subjects'][s]['accuracy_percent']])
 lines=['# 纯RMU 40/80/120步连续轨迹对照','', '本轮唯一变量为训练长度；同一原模型起点、同一条连续120次优化器更新轨迹，固定40/80/120步。hook12，更新10/11/12层down_proj.weight；alpha2、gamma1、steering32、lr5e-5恒定、batch1×累积16、seq256、seed/data_seed42、BF16/SDPA。标准表示MSE目标，retain NLL仅监控。','', '## 全量结果','', '| 对象 | 步数 | Bio % | 比原模型下降/点 | MMLU % | 比原模型损失/点 |','|---|---:|---:|---:|---:|---:|']
 for r in [original,historical,*rows]:lines.append(f"| {r['experiment']} | {r['step']} | {r['bio_percent']:.4f} | {r['bio_drop_pp']:+.4f} | {r['mmlu_percent']:.4f} | {r['mmlu_loss_pp']:+.4f} |")
 lines+=['','Bio1273题，MMLU14042题；0-shot，无chat/system模板，batch1、BF16/SDPA、归档选项loglikelihood最大值评分。全量逐题样本、开发题ID、提示及答案核对均留档。','', '## 历史复现与监控隔离','',f"第40步参数逐位一致：{report['reproduction40']['bitwise_parameter_match']}。索引、320619个input token及319339个有效预测token、控制方向、逐步损失和开发预测均核对（不逐位一致时在数值审查文件记录差异，非完全复现）；复用历史40步全量：{report['reproduction40']['reuse_historical_full']}。复用时复制逐题记录并重新核对答案、得分最大项与汇总算术，不直接复制报告数字。",'', '文章验证沿用0/每10步。首次控制向量初始化消耗历史随机抽样一次，随后验证、保存重建和开发评测均保护Python/NumPy/CPU/CUDA RNG和模式，并核对参数、优化器、调度器、梯度未变。小规模探针在独立进程执行，正式重新设seed并加载基模；最终冻结参数及参考模型指纹一致。','', '## 长度变化与分组','', '| 比较 | Bio变化/点 | MMLU变化/点 | 生物10科变化/点 | 其他47科变化/点 |','|---|---:|---:|---:|---:|']
 for c in changes:lines.append(f"| {c['to']}−{c['from']} | {c['bio_change_pp']:+.4f} | {c['mmlu_change_pp']:+.4f} | {c['biology_change_pp']:+.4f} | {c['other_change_pp']:+.4f} |")
 lines+=['','Bio变化为负表示准确率进一步降低；MMLU变化为负表示能力损失扩大。10/47科沿用既有固定定义，组内按全量题数加权。','', '| 步数 | 生物10科 % | 相对原模型/点 | 其他47科 % | 相对原模型/点 |','|---|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['step']} | {r['biology_related']['weighted_percent']:.4f} | {r['biology_related']['change_pp']:+.4f} | {r['other']['weighted_percent']:.4f} | {r['other']['change_pp']:+.4f} |")
 lines+=['','各步相对40步下降最大的科目（小科目差值噪声较大，不作为稳定学科结论）：']
 for r in rows[1:]:
  worst=sorted(r['subjects'],key=lambda s:r['subjects'][s]['accuracy_percent']-rows[0]['subjects'][s]['accuracy_percent'])[:8]
  lines.append(f"- {r['step']}步："+'；'.join(f"{s.removeprefix('mmlu_')} {r['subjects'][s]['accuracy_percent']-rows[0]['subjects'][s]['accuracy_percent']:+.2f}点" for s in worst))
 lines+=['', '## 本轮候选与取舍','']
 for t in ['2','3']:lines.append(f"- MMLU损失≤{t}点："+('、'.join(f'{s}步' for s in feasible[t]) if feasible[t] else '无')+'。')
 lines+=['- 无约束Bio最低：'+ '、'.join(f'{s}步' for s in report['bio_lowest_steps'])+'。','- 本轮互不支配候选：'+ '、'.join(f'{s}步' for s in pareto)+'。历史其他配置不混入候选。']
 for r in rows:
  if r['dominated_by']:lines.append(f"- {r['step']}步被"+'、'.join(f'{s}步' for s in r['dominated_by'])+'支配。')
 lines+=['', '## 固定开发子集','', '| 对象 | Bio % | MMLU等权 % | MMLU全量科目数加权 % | 加权相对同子集基线/点 |','|---|---:|---:|---:|---:|']
 for r in [devbaseline,*dev]:lines.append(f"| {r['experiment']} | {r['bio_percent']:.4f} | {r['mmlu_equal_percent']:.4f} | {r['mmlu_weighted_percent']:.4f} | {r['mmlu_weighted_percent']-devbaseline['mmlu_weighted_percent']:+.4f} |")
 lines+=['','固定Bio256/MMLU1710题（每科30）；不与全量基线混用，不据此改变时点。','', '## 时间、资源及保存状态','', f"训练总过程{resource['total_training_process_seconds']:.2f}秒；trainer.train墙钟{resource['trainer_wall_seconds_including_callbacks']:.2f}秒（含验证/保存/开发回调及复现审查等待{report_pause:.2f}秒）；逐次优化器更新循环累计{resource['optimizer_update_loop_seconds']:.2f}秒（含前后向、更新和梯度检查，不声称纯GPU内核时间）；回调及完整性检查累计{resource['callback_seconds_including_checks']:.2f}秒。",f"新增全量评分{resource['new_full_inference_seconds']:.2f}秒；评测进程总墙钟{resource['evaluation_wall_seconds_including_loading_and_checks']:.2f}秒（含加载和核验，复用40步不重复计评分）。",f"训练峰值allocated/reserved：{resource['training_peak_allocated_bytes']/2**30:.3f}/{resource['training_peak_reserved_bytes']/2**30:.3f} GiB。本轮增量载荷{resource['new_delta_payload_bytes']/2**20:.2f} MiB；运行目录实际分配{allocated/2**20:.2f} MiB，新增LFS对象实际分配{lfs_alloc/2**20:.2f} MiB。记录时相对开跑的文件系统净占用增加{resource['free_space_change_since_run_start_bytes']/2**30:.3f} GiB，可用{disk.free/2**30:.2f} GiB。",'', 'LFS容量/额度：**未知**。新增9个增量对象仅在本地持久数据盘；没有自动上传，不能声称远端备份。具体路径、SHA和真实大小见运行目录delta_storage_status.json；文本清单及公开逐题结果正常提交推送。原有历史对象未删除。增量不含优化器状态，不能独立精确断点续训。','', '## 复现与局限','',f"实际训练代码提交：`{train['code_commit']}`；源码SHA、依赖、解析配置、实际TrainingArguments、启动命令、采样索引及曝光、分步loss/NLL、重建检查和回调隔离证据均在`{RUN.relative_to(ROOT)}`。",'', '官方test已参与此前探索，本轮为单seed探索性结果；微小差值不能解释为稳定优势。WMDP准确率下降不等同于知识彻底删除。本轮不自动追加训练，也不将120步视为已找到训练上限。', '', '[机器可读JSON](../logs/rmu_step_comparison_results.json)、[全量CSV](../logs/rmu_step_comparison_full.csv)、[开发CSV](../logs/rmu_step_comparison_development.csv)、[逐学科CSV](../logs/rmu_step_comparison_subjects.csv)。']
 (ROOT/'docs/rmu-step-comparison-results.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'full':[{k:r[k] for k in ['step','bio_percent','mmlu_percent','mmlu_loss_pp']} for r in rows],'feasible':feasible,'pareto':pareto,'resources':resource},indent=2))
if __name__=='__main__':main()
