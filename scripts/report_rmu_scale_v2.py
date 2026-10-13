"""Audit bounded scale round and compile Chinese evidence without inference."""
import csv, hashlib, json, shutil, subprocess
from pathlib import Path
from rmu_adaptation_utils import ROOT, REPO, OUT, base_samples, dump, materialized, digest
from reweight_rmu_development import summarize

NEW = REPO/'saves/unlearn/rmu_scale_v2'
ORIGINAL = None
def read(p): return json.loads(p.read_text())
def flat(data):
    return {task: value if isinstance(value,list) else [dict(row,task=name) for name,rows in value.items() for row in rows] for task,value in data.items()}
def full_row(name,data,config):
    data=flat(data); m=summarize(data['mmlu']); metrics={}
    for task,n in [('wmdp_bio',1273),('mmlu',14042)]:
        rows=data[task]; assert len(rows)==n
        assert len({(r['task'],r['doc_id']) for r in rows})==n
        if ORIGINAL is not None:
            reference={(r['task'],r['doc_id']):r for r in ORIGINAL[task]}
            assert {(r['task'],r['doc_id']) for r in rows}==set(reference)
            for r in rows:
                before=reference[(r['task'],r['doc_id'])]
                assert r['doc_hash']==before['doc_hash'] and r['prompt_hash']==before['prompt_hash']
                if 'baseline_acc' in r: assert r['baseline_acc']==before['acc']
        correct=sum(r['acc'] for r in rows)
        metrics[task]={'n':n,'correct':correct,'accuracy_percent':100*correct/n}
    assert abs(m['all']['weighted_percent']-metrics['mmlu']['accuracy_percent'])<1e-10
    return {'name':name,'config':config,'metrics':metrics,'mmlu_subjects':m}

def main():
    global ORIGINAL
    ORIGINAL=flat(base_samples())
    dev=read(ROOT/'logs/rmu_scale_v2_reweighted_development.json'); selection=read(ROOT/'logs/rmu_scale_v2_full_selection.json')
    assert len(selection['selected'])<=2
    baseline=full_row('original',base_samples(),{'hook':None,'alpha':None,'steering':None,'steps':0})
    assert baseline['metrics']['wmdp_bio']['correct']==940 and baseline['metrics']['mmlu']['correct']==9924
    full=[baseline]
    old={}
    for task,folder in [('wmdp_bio','qwen3_4b_rmu_formal80_wmdp_bio_0shot_full_v1'),('mmlu','qwen3_4b_rmu_formal80_mmlu_0shot_full_v1')]:
        old[task]=read(REPO/'saves/eval'/folder/'LMEval_EVAL.json')[task]
    full.append(full_row('legacy_h7_a1_s2_80',old,{'hook':7,'alpha':1,'steering':2,'steps':80}))
    for folder,alpha in [('selected_full_evaluation',1),('selected_full_evaluation_phase2',2)]:
        full.append(full_row(f'legacy_h12_a{alpha}_s2_40',read(OUT/folder/'samples.json'),{'hook':12,'alpha':alpha,'steering':2,'steps':40}))
    for selected in selection['selected']:
        cp=(ROOT/selected['path']).parent.parent; run=read(cp.parent/'run.json'); name=cp.parent.name+'_'+cp.name
        folder=NEW/'full_evaluations'/name; result=read(folder/'result.json')
        row=full_row(name,read(folder/'samples.json'),{k:run[k] for k in ['hook','alpha','steering','lr','steps','seed']})
        assert row['metrics']=={k:{key:v[key] for key in ['n','correct','accuracy_percent']} for k,v in result['metrics'].items()}
        row['evaluation_config']=read(folder/'evaluation_config.json'); full.append(row)
    for row in full:
        for task in row['metrics']: row['metrics'][task]['change_pp']=row['metrics'][task]['accuracy_percent']-baseline['metrics'][task]['accuracy_percent']
        for group in ['biology_related','other']:
            row['mmlu_subjects'][group]['weighted_change_pp']=row['mmlu_subjects'][group]['weighted_percent']-baseline['mmlu_subjects'][group]['weighted_percent']
        row['loss_le_2pp']=row['metrics']['mmlu']['change_pp']>=-2
        row['loss_le_3pp']=row['metrics']['mmlu']['change_pp']>=-3
        row['pareto_full']=not any(q['metrics']['wmdp_bio']['accuracy_percent']<=row['metrics']['wmdp_bio']['accuracy_percent'] and q['metrics']['mmlu']['accuracy_percent']>=row['metrics']['mmlu']['accuracy_percent'] and (q['metrics']['wmdp_bio']['accuracy_percent']<row['metrics']['wmdp_bio']['accuracy_percent'] or q['metrics']['mmlu']['accuracy_percent']>row['metrics']['mmlu']['accuracy_percent']) for q in full)
    trials=[]; cpbytes=0; cpcount=0
    for folder in sorted(NEW.glob('h12_*')):
        run=read(folder/'run.json'); assert run['status']=='completed' and run['steps']==40
        vector=read(folder/'control_vector.json')
        assert vector['unit_direction_sha256']==vector['saved_original_direction_sha256']=='4ce2cff959ea65bf75a4ed391c3db0bfae83aa5f226a110217532b310565a967'
        assert vector['l2_norm']==run['steering']
        consumed=read(folder/'consumed_block_indices.json'); assert consumed==read(OUT/'h12_a1_lr5e-05_s2_screen40/consumed_block_indices.json')
        assert len(consumed['forget'])==len(consumed['retain'])==640
        if run['coverage_dir']:
            for role in ['forget','retain']:
                positions=read(folder/f'{role}_consumed_positions.json')
                assert len(positions)==640 and [p['index'] for p in positions]==consumed[role]
        for key in ['frozen_parameters_unchanged','reference_unchanged','sampling_sequence_identical','control_direction_identical']: assert run[key]
        assert run['actual_input_tokens']==320619 and run['actual_loss_tokens']==319339
        for check in run['development']:
            p=ROOT/check['path']; assert (p/'checkpoint.json').exists()==check['weights_saved']
            if check['weights_saved']:
                assert check['save_reload_identical']; manifest=read(p/'checkpoint.json'); assert len(manifest['parameters'])==3
                for param in manifest['parameters']:
                    payload=materialized(p/param['file']); assert payload.stat().st_size==param['bytes'] and digest(payload.read_bytes())==param['compressed_sha256']
                    assert param['base_tensor_sha256']!=param['trained_tensor_sha256']; cpbytes+=param['bytes']
                cpcount+=1
        if folder.name=='h12_a1_s16_fixed40': assert run['saved_steps']==[40]
        else: assert run['saved_steps']==[10,20,40]
        trials.append(run)
    assert len(trials)==4 and cpcount==10
    space={name:{'total':s.total,'used':s.used,'free':s.free} for name,path in [('data',ROOT),('system',Path('/tmp'))] for s in [shutil.disk_usage(path)]}
    assert space['data']['free']>1073741824
    audit={'trials_completed':4,'new_hyperparameter_groups':3,'data_coverage_groups':1,'new_saved_checkpoints':cpcount,'actual_compressed_delta_bytes':cpbytes,'full_evaluations':len(selection['selected']),'space':space,'first_group_only_final_weights':True,'all_saved_reload_identical':True,'all_compressed_objects_sha256_verified':True,'matched_token_and_article_budget':True,'full_question_and_prompt_hashes_identical':True,'optimizer_state_saved':False,'independent_test':False,'remote_publication_requested':False,'report_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    dump(ROOT/'logs/rmu_scale_v2_completion_audit.json',audit)
    dump(ROOT/'logs/rmu_scale_v2_full_comparison.json',{'rows':full,'exploratory':True,'biology_subject_definition':dev['biology_subject_definition']})
    columns=['name','hook','updated_parameters','lr','alpha','steering','steps','bio_percent','bio_change_pp','mmlu_percent','mmlu_change_pp','biology_change_pp','other_change_pp','pareto_full','loss_le_2pp','loss_le_3pp']
    with (ROOT/'logs/rmu_scale_v2_full_comparison.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n'); w.writerow(columns)
        for r in full:
            c=r['config']; h=c['hook']; w.writerow([r['name'],h,','.join(f'model.layers.{i}.mlp.down_proj.weight' for i in range(h-2,h+1)) if h else '',c.get('lr',5e-5) if h else '',c['alpha'],c['steering'],c['steps'],r['metrics']['wmdp_bio']['accuracy_percent'],r['metrics']['wmdp_bio']['change_pp'],r['metrics']['mmlu']['accuracy_percent'],r['metrics']['mmlu']['change_pp'],r['mmlu_subjects']['biology_related']['weighted_change_pp'],r['mmlu_subjects']['other']['weighted_change_pp'],r['pareto_full'],r['loss_le_2pp'],r['loss_le_3pp']])
    lines=['# RMU 第二轮尺度与覆盖适配：探索性结果','',
    '本轮完成先诊断、开发分数重算、3组尺度/保留权重训练和1组覆盖对照，最后按预先锁定规则执行2次全量评测。实际模型为 Qwen3-4B-Instruct-2507；所有候选从同一个原始模型开始。旧模型、失败结果和日志均保留；本轮只提交本地 Git，不推送或发布模型。','',
    '## 诊断及实现证据','',
    'hook12 是从0开始的第13个 decoder block，捕获整个 block 输出。只更新10/11/12层的 mlp.down_proj.weight，合计74,711,040参数；参考模型冻结且eval/no_grad，其他参数指纹保持不变。训练模型train，诊断与选择题评测eval。MSE先对隐藏维平均、再对每篇有效token平均、最后对文章平均，总损失为forget MSE + alpha × retain MSE。attention mask与labels=-100排除无效token；语言模型retain NLL另按移位标签计算。训练输入为原生token块，配置中的聊天模板字段不实际作用于这些预先tokenized块。','',
    '原始forget表示平均L2=44.837、median=44.576，retain平均44.378；旧steering2仅为forget平均范数的4.46%。控制向量原为正值torch.rand后L2归一化，本轮从旧保存向量恢复固定方向，再乘16或32（约原范数36%/71%），不是重新抽目标，也没有照搬steering20/alpha100。','',
    '诊断使用按文章分组独立划出的64块/角色，排除padding。分布、分位数、每篇MSE及固定文章0/16/32/48的两部分梯度见 logs/rmu_scale_v2_diagnostics.json 与 postdiagnostics.json。旧alpha1/alpha2的forget范数约29.278/30.885、与原表示cos约0.860/0.897：既压缩范数又改变方向。原始retain梯度为0，比例记null。前诊断原始angular MSE约−5e−8来自余弦浮点舍入；后诊断将cos夹在[-1,1]，不改动原始记录。','',
    '| 后诊断模型 | forget L2 | retain L2 | forget cos原始 | forget cos目标 | forget MSE | alpha×retain MSE |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    post=read(ROOT/'logs/rmu_scale_v2_postdiagnostics.json')
    for name,r in post['results'].items():
        f=r['roles']['forget']; t=r['roles']['retain'];lines.append(f"| {name} | {f['l2_norm']['mean']:.3f} | {t['l2_norm']['mean']:.3f} | {f['cosine_to_original']['mean']:.4f} | {f['cosine_to_saved_target']['mean']:.4f} | {f['article_mean_mse_native_bf16']:.4f} | {r['alpha']*t['article_mean_mse_native_bf16']:.4f} |")
    lines+=['','| 梯度诊断模型 | forget梯度L2范围 | alpha×retain梯度L2范围 | forget/(alpha×retain)范围 |','| --- | ---: | ---: | ---: |']
    pre=read(ROOT/'logs/rmu_scale_v2_diagnostics.json')
    for prefix,diag in [('旧',pre),('新',post)]:
        for name,r in diag['results'].items():
            probes=r.get('gradient_probes',r.get('gradients',[]))
            if not probes: continue
            f=[p['forget_gradient_norm'] for p in probes];t=[p['alpha_weighted_retain_gradient_norm'] for p in probes];ratio=[p['forget_over_alpha_retain_ratio'] for p in probes if p['forget_over_alpha_retain_ratio'] is not None]
            lines.append(f"| {prefix}/{name} | {min(f):.4f}–{max(f):.4f} | {min(t):.4f}–{max(t):.4f} | {f'{min(ratio):.3f}–{max(ratio):.3f}' if ratio else 'null'} |")
    lines+=['','新steering32/alpha2的forget cos目标显著增加，范数仍低于原始；范数与方向都发生变化。这些表示和梯度证据不能等同于知识删除。四个probe只是局部梯度诊断，不能代表整个训练梯度分布。','',
    '## 预先计划及公平性','',
    '固定hook12、10/11/12 down_proj、lr5e-5、40步、seed42、batch1×累积16、256token。同方向及文章顺序均核实。先alpha1/steering16，仅40步保存权重；10/20步仅存指标。复核空间后运行alpha1/steering32，其开发MMLU损失3.17点触发预先条件，再仅增alpha至2。每组40步，无多因素搜索。各组实际input token320619、loss token319339。','',
    '覆盖组只改变块位置，仍用alpha1/steering2，与旧固定块40步对照。相同2048训练文章/角色和640次输入/角色；所有验证文章组排除，不加入评测题。每篇最多4个不重叠位置，hash决定首访及循环；逐样本长度、采样顺序、实际token预算严格相同。forget不同位置640→640、retain545→631，非原位置输入636/553次。forget每篇本轮只访问一次，因此这是位置替换与retain重复访问多样化，不能声称完整文章覆盖或更大总token预算。数据版本和实际位置见coverage_manifest、coverage_readiness及run目录；受限token文件留本地并被Git忽略。','',
    '## 开发集：全体候选','',
    'Bio固定256题；MMLU每科30题共1710题。原始MMLU等权73.5088%、按全量各科题数加权69.9841%；所有候选用同一组57科权重，总14042。加权修正科目组成差异，不能消除30题/科的小样本误差；等权和加权排名都可能与全量不同，不将细微差距作强结论。开发题来自已经观察过的官方test，不是独立验证集；retain训练语料也不是能力验证集。','',
    '| 实验/步 | hook | alpha | steering | Bio% | MMLU等权% | MMLU加权% | 加权Δ点 | Pareto | 权重可用 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    dev_csv=[]
    baseline_dev_bio=next(r['bio_percent'] for r in dev['rows'] if 'original_development' in r['path'])
    for r in dev['rows']:
        p=ROOT/r['path']; orig='original_development' in str(p)
        if orig: name='original';h=a=s='—';step=0;available=True
        else:
            folder=p.parent.parent.parent; c=read(folder/'training_config.json'); name=folder.name;step=int(p.parent.parent.name.split('-')[1]); available=(p.parent.parent/'checkpoint.json').exists()
            # RMU config names vary across historical runners; folder regex is authoritative for displayed names.
            import re
            h=int(re.search(r'h(\d+)',name)[1]);a=float(re.search(r'_a([\d.]+)',name)[1]);s=float(re.search(r'_s([\d.]+)',name)[1])
        lines.append(f"| {name}/{step} | {h} | {a} | {s} | {r['bio_percent']:.4f} | {r['mmlu']['all']['equal_percent']:.4f} | {r['mmlu']['all']['weighted_percent']:.4f} | {r['weighted_change_pp']:+.4f} | {'是' if r['pareto_weighted'] else '否'} | {'是' if available else '否（仅指标）'} |")
        lr=2.5e-5 if 'lr2.5e-05' in name else 5e-5
        params=','.join(f'model.layers.{i}.mlp.down_proj.weight' for i in range(h-2,h+1)) if isinstance(h,int) else ''
        dev_csv.append([name,step,h,params,lr if not orig else '',a,s,r['bio_percent'],r['bio_percent']-baseline_dev_bio,r['mmlu']['all']['equal_percent'],r['mmlu']['all']['weighted_percent'],r['equal_change_pp'],r['weighted_change_pp'],r['mmlu']['biology_related']['weighted_change_pp'],r['mmlu']['other']['weighted_change_pp'],r['pareto_weighted'],available])
    with (ROOT/'logs/rmu_scale_v2_development_config_comparison.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['experiment','step','hook','updated_parameters','lr','alpha','steering','bio_percent','bio_change_pp','mmlu_equal_percent','mmlu_weighted_percent','mmlu_equal_change_pp','mmlu_weighted_change_pp','biology_change_pp','other_change_pp','pareto_weighted','weights_available']);w.writerows(dev_csv)
    lines+=['','所有表中训练候选参数类型为down_proj.weight，更新层为hook前两层及自身；学习率默认5e-5，名称lr2.5e-05的历史组为2.5e-5，其余保持不变。完整逐科、生物相关组及其他组变化见 reweighted_development.json 与CSV。','']
    for threshold,r in dev['best_at_2_and_3pp'].items():lines.append(f"- 加权MMLU损失≤{threshold}点最佳开发候选：`{Path(r['path']).parent.parent.parent.name}`，Bio {r['bio_percent']:.4f}%，MMLU加权变化{r['weighted_change_pp']:+.4f}点。")
    lines+=['','两档最佳相同，因此第二个全量点预先选更低Bio的alpha1/steering32，开发损失3.17点，明确在3点之外，不将其伪装为满足约束；最多2次全量，选择文件在全量开始前已提交。覆盖组Bio64.8438%、加权MMLU损失1.9499点，未成为Pareto点，也未见明确覆盖收益。','',
    '## 全量：探索性比较','',
    '1273题Bio、14042题MMLU；逐题复用原始归档提示及4个选项评分请求，0-shot、无chat/system、BF16/SDPA、batch1，loglikelihood argmax。全量MMLU按题数汇总，与57科按正式题数加权严格相等。官方test早已参与探索，下表不是独立测试。','',
    '| 模型 | Bio% | BioΔ点 | MMLU% | MMLUΔ点 | ≤2点 | ≤3点 | 全量Pareto |','| --- | ---: | ---: | ---: | ---: | --- | --- | --- |']
    for r in full:
        b=r['metrics']['wmdp_bio'];m=r['metrics']['mmlu'];lines.append(f"| {r['name']} | {b['accuracy_percent']:.4f} | {b['change_pp']:+.4f} | {m['accuracy_percent']:.4f} | {m['change_pp']:+.4f} | {'是' if r['loss_le_2pp'] else '否'} | {'是' if r['loss_le_3pp'] else '否'} | {'是' if r['pareto_full'] else '否'} |")
    lines+=['','| 全量模型 | 生物相关10科Δ点 | 其他47科Δ点 |','| --- | ---: | ---: |']
    for r in full:lines.append(f"| {r['name']} | {r['mmlu_subjects']['biology_related']['weighted_change_pp']:+.4f} | {r['mmlu_subjects']['other']['weighted_change_pp']:+.4f} |")
    lines+=['','生物相关定义预先固定为anatomy、clinical_knowledge、college_biology、college_medicine、high_school_biology、human_aging、medical_genetics、nutrition、professional_medicine、virology，覆盖2094题；是宽泛生物医学分组，不是官方领域标签。其他47科11948题。','']
    for t in [2,3]:
        best=min((r for r in full if r['loss_le_'+str(t)+'pp']),key=lambda r:(r['metrics']['wmdp_bio']['accuracy_percent'],-r['metrics']['mmlu']['accuracy_percent']))
        lines.append(f"在已观测全量结果中，损失≤{t}点的最低Bio为 `{best['name']}`：Bio{best['metrics']['wmdp_bio']['accuracy_percent']:.4f}%，MMLU{best['metrics']['mmlu']['accuracy_percent']:.4f}%。这是探索性比较，不新增调参、不将3点参考阈值解释为遗忘成功。")
    newly_evaluated=full[4:]
    if not any(r['loss_le_2pp'] for r in newly_evaluated):
        lines+=['','本轮新增全量候选均未满足原来的2点约束；2点档仍保留旧alpha2/steering2候选，不能把开发达标冒充全量达标。']
    for r in newly_evaluated:
        old_a2=full[3];b=r['metrics']['wmdp_bio'];m=r['metrics']['mmlu']
        lines.append(f"新增 `{r['name']}` 相比旧hook12/alpha2/steering2：Bio变化{b['accuracy_percent']-old_a2['metrics']['wmdp_bio']['accuracy_percent']:+.4f}点，MMLU变化{m['accuracy_percent']-old_a2['metrics']['mmlu']['accuracy_percent']:+.4f}点。该变化代表不同取舍，是否全面改进由上表Pareto标记判断。")
    lines+=['','## 运行、保存和限制','',
    '| 新实验 | 代码提交 | 运行秒数* | 峰值allocated GiB | reserved GiB | 保存权重步 |','| --- | --- | ---: | ---: | ---: | --- |']
    for r in trials:lines.append(f"| {r['name']} | {r['code_commit'][:7]} | {r['elapsed_seconds']:.1f} | {r['cuda_peak_allocated_bytes']/2**30:.3f} | {r['cuda_peak_reserved_bytes']/2**30:.3f} | {r['saved_steps']} |")
    lines+=['','*运行秒数包括训练、固定文章验证及开发评测/保存开销；纯Trainer train_runtime详见run.json。完整配置、数据manifest、控制向量、训练/验证逐步损失、实际采样索引和位置均留档。PyTorch2.9.1+cu128、Transformers5.5.4、lm_eval0.4.11、datasets3.0.1。模型本地文件SHA及完整原始参数指纹固定，远端模型immutable revision未取得，不虚构版本。诊断与评测脚本均留档。','',
    f"数据盘60GiB，初始可用{read(ROOT/'logs/rmu_scale_v2_space_plan.json')['data_disk']['free']/2**30:.3f}GiB，完成审计时剩余{space['data']['free']/2**30:.3f}GiB；系统盘剩余{space['system']['free']/2**30:.3f}GiB。10个新增检查点实际压缩对象{cpbytes/2**20:.3f}MiB，非pointer大小；均保存在数据盘.git/lfs/objects，不依赖/tmp唯一副本。原计划含临时/LFS/日志/评测/Git总保守预算约2.1GiB、至少1GiB余量；每次gate按实际空间复核。没有删除历史模型、LFS对象或扩容付费。",'',
    '检查点为BF16逐位XOR+gzip，必须使用固定原始模型和load_delta重建，不是LoRA或独立HF完整模型。保存后已重载核实逐位一致，最终评测另从完整指纹一致的原模型重建。未保存优化器状态，不能宣称可断点续训。失败和负面结果不覆盖。','',
    '本轮只一个seed、40步有限预算，覆盖一次文章位置、候选少量；80步首轮和40步适配的差异不能全部归于换层或steering。Bio选择题下降只表示该代理任务表现下降，不证明危险知识彻底删除、不可恢复或系统安全。后续优先对较好取舍做少量预先声明的多seed复验与恢复测试；本轮没有自动执行。标准平台交付包仍须另按Word要求整理，受限训练正文/数据不得公开发布。','',
    '证据：`logs/rmu_scale_v2_full_comparison.json/.csv`、`logs/rmu_scale_v2_completion_audit.json`、`logs/rmu_scale_v2_full_selection.json`、`docs/rmu-scale-v2-plan.md`；所有新增模型和逐题结果位于 `open-unlearning/saves/unlearn/rmu_scale_v2/`。']
    lines+=['','补充来源：清洗后forget训练/验证23220/1217条、retain57488/3029条，按文章组划分。formal_blocks_v1每角色2048训练候选块及64验证块，最长256token、短文原长，不跨文章拼接。本轮不重新划分文章。两份诊断执行时脚本为新增工作树文件；`logs/rmu_scale_v2_diagnostic_provenance.json`记录脚本SHA与后来保存完全相同内容的提交，执行时HEAD不单独充当完整源码快照。']
    (ROOT/'docs/rmu-scale-v2-results.md').write_text('\n'.join(lines)+'\n')
    print('AUDIT_COMPLETE',audit);print('FULL',[(r['name'],r['metrics']) for r in full])

if __name__=='__main__':main()
