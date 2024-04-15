from entailment_bank.utils.nlp_agent import MultiAngleModel, NlpAgent
from llama_entailer import llama_Entailer
from entailer import Entailer
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json

# truth_device = 'cuda:7'

# truth_judge = AutoModelForCausalLM.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B").to(truth_device)
# tokenizer = AutoTokenizer.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B", max_length=100)
ew_model = MultiAngleModel(model_path="allenai/entailer-11b", cuda_devices=[10, 11])
prover = NlpAgent(model=ew_model, default_outputs="proof")
entail_verifier = NlpAgent(model=ew_model, default_outputs=["implied"], default_options={"explicit_outputs": ['true', 'false']})
hyp_verifier = NlpAgent(model=ew_model, default_outputs=["valid"], default_options={"explicit_outputs": ['true', 'false']})


entailer = Entailer(ew_model, prover, entail_verifier, hyp_verifier)

# Testing on obq dataset
dataset = load_dataset("allenai/openbookqa")
crct=0

with open("fulldepth_obqa.jsonl", 'a') as f:
    for i in tqdm(range(len(dataset['test']))):
        data = dataset["test"][i]
        best_score, best_tree = 0, {}
        best_choice = "A"
        choices = {}
        for i in range(len(data["choices"]["text"])):
            hyp = data["question_stem"]+" "+data["choices"]['text'][i]
            score, tree, prem_scores, entail_score = entailer.generate_entailment_tree(hyp, {}, 3, 1, full_depth=True)
            
            choices[data["choices"]["label"][i]] = {"premises": prem_scores, "entail_score": entail_score}
            
            if score>best_score:
                best_score = score
                best_tree = tree
                best_choice = data["choices"]["label"][i]
                
            json.dump({"HYP": hyp, "tree": tree}, f)
            f.write('\n')
        
        if best_choice==data["answerKey"]:
            crct+=1

obqa_accuracy = crct/len(dataset["test"]["question_stem"])
print("Accuracy on OBQA dataset: ", obqa_accuracy)



# Testing on quartz dataset
dataset = load_dataset("allenai/quartz")
crct=0

with open('fulldepth_quartz.jsonl','a') as f:
    for i in tqdm(range(len(dataset['test']))):
        data = dataset["test"][i]
        best_score, best_tree = 0, {}
        best_choice = "A"
        choices = {}
        for j in range(len(data["choices"]["text"])):
            hyp = data["question"]+" "+data["choices"]['text'][j]
            score, tree, prem_scores, entail_score = entailer.generate_entailment_tree(hyp, {}, 2, 1, prover_prefix = data["para"], full_depth=True)
            
            choices[data["choices"]["label"][j]] = {"premises": prem_scores, "entail_score": entail_score}
            
            if score>best_score:
                best_score = score
                best_tree = tree
                best_choice = data["choices"]["label"][j]
        
        if best_choice==data["answerKey"]:
            crct+=1
        
            json.dump({"HYP": hyp, "tree": tree}, f)
            f.write('\n')
        
            
    quartz_accuracy = crct/len(dataset["test"]["question"])
    print("Accuracy on quartz dataset: ", quartz_accuracy)

with open('llama_entailer_result.txt', 'a') as f:
    f.write(f'Accuracy on obqa dataset: {obqa_accuracy}\n')
    f.write(f'Accuracy on quartz dataset: {quartz_accuracy}\n')
    