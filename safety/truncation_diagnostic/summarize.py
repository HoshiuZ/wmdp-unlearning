"""No human safety annotation: objective endings, decoded prefixes, text heuristics and Guard labels."""
import re,os,statistics
from collections import Counter
from .common import *

def features(text):
 words=re.findall(r'\S+',text);grams=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
 repetition=1-len(set(grams))/len(grams) if grams else 0
 lines=[s.strip() for s in text.splitlines() if s.strip()];run=maxrun=1
 for a,b in zip(lines,lines[1:]):
  run=run+1 if a==b and len(a)>=20 else 1;maxrun=max(maxrun,run)
 return {'unclosed_code_fence':text.count('```')%2==1,'ends_with_connective':bool(re.search(r'\b(and|or|the|to|of|with|that|a|an|in)\s*$',text,re.I)),
         'nonterminal_tail':bool(text.strip() and text.rstrip()[-1] not in '.!?。！？”\"\'）)]}'),
         'high_fourgram_repetition':len(words)>=80 and repetition>.5,'fourgram_repetition_fraction':repetition,
         'repeated_long_lines':maxrun>=3,'format_control_or_think':any(s in text for s in ['<think>','</think>','<|im_start|>','<|im_end|>']),
         'replacement_character':'\ufffd' in text}

def main():
 c=connection();m=manifest();selected=rows();assert len(selected)==40
 result={'protocol_sha256':m['sha256'],'n_requests':40,'input_labels':dict(Counter(r['gold'] for r in selected)),'original_endings':m['original_endings'],'models':{},'sessions':{},'prefix_caveat':m['prefix_caveat']};private=[]
 for model in ('original','rmu80'):
  counter=Counter();prefix_lengths=[];prefix_fractions=[];trans=Counter();flags={v:Counter() for v in ('512','1024')};old_labels=Counter();new_labels=Counter();errors=Counter()
  for row in selected:
   a=oldgen(row,model);b=get(c,key('generate1024',model,row));assert b and b['status']=='ok';assert sha(b['response'].encode())==b['response_sha256'];assert len(b['token_ids'])==b['output_tokens']<=1024
   assert sha(a['response'].encode())==a['response_sha256']
   common=len(os.path.commonprefix([a['response'],b['response']]));prefix_lengths.append(common);prefix_fractions.append(common/len(a['response']))
   old_cut=not a['eos'] and a['output_tokens']==512;counter['old_cutoff']+=old_cut;counter['new_natural']+=b['eos'];counter['new_cutoff']+=b['cutoff'];counter['new_natural_at_1024']+=bool(b['eos'] and b['output_tokens']==1024)
   exact=b['response'].startswith(a['response']);decoded=b['prefix_at_old_token_count']==a['response'];counter['exact_string_prefix']+=exact;counter['same_decoded_prefix_at_old_token_count']+=decoded
   if old_cut:counter['old_cutoff_now_natural']+=b['eos'];counter['old_cutoff_exact_string_prefix']+=exact;counter['old_cutoff_same_decoded_prefix']+=decoded
   else:counter['old_natural_same_complete_text']+=a['response']==b['response']
   counter['old_output_tokens']+=a['output_tokens'];counter['new_output_tokens']+=b['output_tokens']
   aa=get(c,key('audit512',model,row));bb=get(c,key('audit1024',model,row));assert aa is not None and bb is not None
   valid=lambda v:v['status']=='ok' and v['parse_ok']
   for length,v in [('512',aa),('1024',bb)]:
    if not valid(v):errors[length]+=1
   old_labels[aa['label']]+=1;new_labels[bb['label']]+=1;trans[(aa['label'],bb['label'])]+=1
   counter['guard_label_changed']+=aa['label']!=bb['label'];counter['main_action_changed']+=aa['policies']['main']['blocked']!=bb['policies']['main']['blocked'];counter['strict_action_changed']+=aa['policies']['strict']['blocked']!=bb['policies']['strict']['blocked']
   fs={str(n):features(v['response']) for n,v in [(512,a),(1024,b)]}
   for length,v in fs.items():flags[length].update({name:int(value) for name,value in v.items() if isinstance(value,bool)})
   private.append({'sample_id':row['sample_id'],'model':model,'input_label':row['gold'],'old_cutoff':old_cut,'old':a,'new':b,'audit512':aa,'audit1024':bb,'heuristics':fs,'prefix_string_exact':exact,'prefix_decoded_exact':decoded})
  result['models'][model]={'n':40,**counter,'common_prefix_characters_median':statistics.median(prefix_lengths),'common_prefix_fraction_median':statistics.median(prefix_fractions),'first100_characters_equal':sum(n>=100 for n in prefix_lengths),'first500_characters_equal':sum(n>=500 for n in prefix_lengths),'old_labels':dict(old_labels),'new_labels':dict(new_labels),'label_transitions':[{"from":a,"to":b,"n":n} for (a,b),n in sorted(trans.items())],'audit_errors':dict(errors),'text_heuristics':{k:dict(v) for k,v in flags.items()}}
  result['sessions'][model]=json.loads((LOCAL/f'{model}_session.json').read_text())
 result['sessions']['guard']=json.loads((LOCAL/'audit_session.json').read_text());result['after_resources']=resource();result['local_allocated_bytes']=sum(p.stat().st_blocks*512 for p in LOCAL.rglob('*') if p.is_file())
 target=LOCAL/'comparison.jsonl';target.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in private));result['comparison_manifest_sha256']=sha(target.read_bytes());result['after_resources']=resource();result['local_allocated_bytes']=sum(p.stat().st_blocks*512 for p in LOCAL.rglob('*') if p.is_file())
 dump(HERE/'results.json',result)
 lines=['# 512→1024 token截断诊断','', '日期：2026-10-15（北京时间）。本次仅40条诊断，不改变原正式回答，不全量重生成，不训练、不开展人工标注或独立安全评判。执行前未找到已完成诊断；运行器若发现本报告与results.json已存在则直接退出，不重复推理。','',
        '## 固定抽样与设置','',f"从任一模型实际无EOS且生成512 token的908条请求并集中，用seed42对有害/正常各随机抽20条，再恢复原请求顺序。样本及标签分布、固定ID、源模型与协议指纹见manifest.json；原始请求与回答只保存在cache/truncation_diagnostic_v1。抽样清单SHA `{m['sample_manifest_sha256']}`。",'',
        '仅max_new_tokens从512变为1024，其他CFG与正式四组相同：官方聊天模板、无system、非思考、do_sample=false、num_beams=1、use_cache=true、batch20、seed42、BF16/SDPA。顺序加载原模型、从既有XOR增量恢复且核对398项指纹的RMU80，再加载固定Guard。无新完整权重副本。完整生成配置、软件版本、耗时和源文件SHA见results.json/sessions。', '',
        '批大小未变，但40条抽样的批内组合与历史正式生成不同，BF16推理的数值差异可能改变greedy前缀。比较旧原文是否为新原文前缀，并比较新生成前原token数解码文本与旧文本；旧记录未保存原始token IDs，因此不声称原始token IDs逐位一致。前缀不一致时不能只归因于上限变化。','',
        '## 正式512结束原因核实','', '|模型|自然结束|无EOS被512截停|恰在512自然结束（包含于自然结束）|','|---|---:|---:|---:|']
 for model,v in m['original_endings'].items():lines.append(f"|{model}|{v['natural']}|{v['cutoff']}|{v['natural_at_512']}|")
 lines+=['','## 40条生成对照','', '|模型|旧版本截停数|1024自然结束|1024仍截停|旧截停→自然|完整字符串前缀一致|前原token数解码一致|','|---|---:|---:|---:|---:|---:|---:|']
 for model,v in result['models'].items():lines.append(f"|{model}|{v['old_cutoff']}|{v['new_natural']}|{v['new_cutoff']}|{v['old_cutoff_now_natural']}|{v['exact_string_prefix']}/40|{v['same_decoded_prefix_at_old_token_count']}/40|")
 lines+=['', '严格前缀配对没有成立：原模型/RMU完整前缀仅2/40、3/40一致；共同字符前缀中位数分别258、338，前100字符一致32/40、34/40。虽然实际GenerationConfig仅max_new_tokens改变，但抽样批内组合不同是未控制因素，不能把后续内容或Guard标签变化单独归因于新增尾部。', '', '## 可复现文本检查（不是人工标注）','', '自动检查未闭合代码围栏、连接词结尾、非句末符结尾、4-gram重复率>50%（至少80词）、连续重复长行≥3、思考/角色控制标记及替换字符。前几项只能提示潜在不完整或格式问题，列表/代码可合法无句末符，不能把提示直接当成确定错误。','', '|模型/上限|未闭合围栏|连接词结尾|非句末符|高重复|重复长行|控制/思考标记|替换字符|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for model,v in result['models'].items():
  for n,f in v['text_heuristics'].items():lines.append(f"|{model}/{n}|"+'|'.join(str(f.get(k,0)) for k in ['unclosed_code_fence','ends_with_connective','nonterminal_tail','high_fourgram_repetition','repeated_long_lines','format_control_or_think','replacement_character'])+'|')
 lines+=['','## Guard双版本输出审核','', '固定Guard revision '+m['guard_revision']+'，官方user请求＋assistant回答模板、既有配置与解析器不变。已有512输出审核经渲染哈希核对后复用；原正式流程中输入被拦而未输出审核的512回答，本次为诊断补审。两个版本均覆盖所有40条，不受输入拦截影响；error/unknown单列、不当Safe。两种策略是本实验动作映射，不是官方策略名称。','', '|模型|512标签|1024标签|等级改变|主策略动作改变|严格策略动作改变|审核异常|','|---|---|---|---:|---:|---:|---|']
 for model,v in result['models'].items():lines.append(f"|{model}|{v['old_labels']}|{v['new_labels']}|{v['guard_label_changed']}|{v['main_action_changed']}|{v['strict_action_changed']}|{v['audit_errors']}|")
 lines+=['','等级转移详见results.json；它们仅用于截断诊断，不是独立安全评判，不能当成安全提升。','', '## 资源','', '|阶段|wall秒|峰值allocated GiB|峰值reserved GiB|','|---|---:|---:|---:|']
 for name,v in result['sessions'].items():lines.append(f"|{name}|{v['wall_seconds']:.2f}|{v['peak_allocated_bytes']/2**30:.3f}|{v['peak_reserved_bytes']/2**30:.3f}|")
 free=result['after_resources']['disk_free_bytes'];delta=m['before_resources']['disk_free_bytes']-free
 lines+=['',f"生成阶段包含加载/指纹及生成，Guard从加载后开始计时；不是在线防护延迟。磁盘净变化{delta/2**20:.2f} MiB，私有目录占用{result['local_allocated_bytes']/2**20:.2f} MiB，剩余{free/2**30:.2f} GiB。无新基准、模型下载或训练。",'', '## 建议','',
 '512上限明显限制了正式回答的自然完成：全量约半数被截停；本次原模型36条旧截停请求中20条在1024自然结束，RMU35条中18条自然结束，未闭合围栏及连接词结尾提示也减少。未检出上述规则定义的高重复或控制/思考格式异常，但非句末符提示不能当确定错误。建议正式独立评判若以完整回答为目标，应提高预算，1024可作候选，保留原512作为固定预算对照。', '', '不过本轮多数前缀发生分歧，不能当作严格的同一回答续写，也不能将等级改变全部归于512之后的新增内容。在决定统一重生成前，应先控制历史批内组合/数值前缀的一致性，再固定新预算；这是后续建议，本次不追加控制实验、不全量重跑。1024仍截停16/40、17/40，故不能声称1024已经足够。样本来自截停并集、正常/有害各20条，不代表全量1024截停率或安全准确率。' , '',
 '已完成本次40条诊断后停止。原四组结果不覆盖；没有人工安全标签或独立评判。']
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT_READY')
if __name__=='__main__':main()
