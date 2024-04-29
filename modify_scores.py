from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from tqdm import tqdm

def faithfulness_score(hyp, premises_dict_list, info_judge, info_tokenizer, device):
    entail_score = 1
    for premise in premises_dict_list:
        premise = premise["HYP"]
        prompt = "Q: "+hyp+"\nA: "+premise+"\nHelpful:"
        inputs = info_tokenizer([prompt], return_tensors="pt").to(device)
        outputs = info_judge.generate(**inputs, output_scores=True, return_dict_in_generate=True, max_length=500)
        input_length = 1 if info_judge.config.is_encoder_decoder else inputs.input_ids.shape[1]
        transition_scores = info_judge.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        generated_tokens = outputs.sequences[:,input_length:]
        score = transition_scores[0][0]
        score = np.exp(score.numpy(force=True))
        if info_tokenizer.decode(generated_tokens[0][0])=='yes':
            entail_score *= score
        else:
            entail_score *= (1.0-score)
    return entail_score

def truthfulness_score(prompt, truth_judge, truth_tokenizer, device):
    prompt = prompt+"\nTrue:"
    inputs = truth_tokenizer([prompt], return_tensors="pt").to(device)
    outputs = truth_judge.generate(**inputs, output_scores=True, return_dict_in_generate=True, max_length=500)
    input_length = 1 if truth_judge.config.is_encoder_decoder else inputs.input_ids.shape[1]
    transition_scores = truth_judge.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    generated_tokens = outputs.sequences[:,input_length:]
    score = transition_scores[0][0]
    score = np.exp(score.numpy(force=True))
    if truth_tokenizer.decode(generated_tokens[0][0])=='yes':
        return score
    return 1.0-score

def modify_faith_scores(info, info_judge, info_tokenizer, device):
    if len(info["premises"])==0:
        return info
    
    info["llama_faith"] = str(faithfulness_score(info["HYP"], info["premises"], info_judge, info_tokenizer, device))
    
    for i in range(len(info["premises"])):
        info["premises"][i] = modify_faith_scores(info["premises"][i], info_judge, info_tokenizer, device)
    
    return info

def modify_truth_scores(info, truth_judge, truth_tokenizer, device):
    if len(info["premises"])==0:
        return info
    
    info["llama_truth"] = str(truthfulness_score(info["HYP"], truth_judge, truth_tokenizer, device))
    
    if not isinstance(info["llama_faith"], str):
        info["llama_faith"] = str(info["llama_faith"])
    
    for i in range(len(info["premises"])):
        info["premises"][i] = modify_truth_scores(info["premises"][i], truth_judge, truth_tokenizer, device)
        
    return info
    

truth_judge = AutoModelForCausalLM.from_pretrained("truthfulqa_reeval/output/llama2_7B_truth_judge_final").to('cuda:12')
truth_tokenizer = AutoTokenizer.from_pretrained("truthfulqa_reeval/output/llama2_7B_truth_judge_final", max_length=500)

modified_lines = []
with open('fulldepth_obqa_scores.jsonl', 'r') as f:
    lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = eval(lines[i])
        line = modify_truth_scores(line, truth_judge, truth_tokenizer, device='cuda:12')
        modified_lines.append(line)   

with open('fulldepth_obqa_scores_arc_wt_finetuned.jsonl', 'a') as f:
    for line in modified_lines:
        json.dump(line, f)
        f.write('\n')


modified_lines = []
with open('fulldepth_quartz_scores.jsonl', 'r') as f:
    lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = eval(lines[i])
        line = modify_truth_scores(line, truth_judge, truth_tokenizer, device='cuda:12')
        modified_lines.append(line)   


with open('fulldepth_quartz_scores_arc_wt_finetuned.jsonl', 'a') as f:
    for line in modified_lines:
        json.dump(line, f)
        f.write('\n')
 

modified_lines = []
with open('fulldepth_truthfulqa_scores.jsonl', 'r') as f:
    lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = eval(lines[i])
        line = modify_truth_scores(line, truth_judge, truth_tokenizer, device='cuda:12')
        modified_lines.append(line)   

with open('fulldepth_truthfulqa_scores_arc_wt_finetuned.jsonl', 'a') as f:
    for line in modified_lines:
        json.dump(line, f)
        f.write('\n')


