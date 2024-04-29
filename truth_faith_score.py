from entailment_bank.utils.nlp_agent import MultiAngleModel, NlpAgent
from llama_entailer import llama_Entailer
from entailer import Entailer
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from datasets import load_dataset

def get_scores(tree, entailer:Entailer, llama_entailer:llama_Entailer):    
    if len(tree.keys()) == 0:
        return []
    res = []
    premises = tree.keys()
    for premise in premises:
        info = {}
        info["HYP"] = premise
        info["t5_truth"] = str(entailer.truthfulness_score(premise))
        info["llama_truth"] = str(llama_entailer.truthfulness_score(premise))
        if len(tree[premise].keys())!=0:
            info["t5_faith"] = str(entailer.faithfulness_score(premise, tree[premise].keys()))
            info["llama_faith"] = str(llama_entailer.faithfulness_score(premise, tree[premise].keys()))
        else:
            info["t5_faith"] = str(0.0)
            info["llama_faith"] = str(0.0)
        
        info["premises"] = get_scores(tree[premise], entailer, llama_entailer)
        
        res.append(info)
        
    return res

def get_score_tree(hyp, entailer:Entailer, llama_entailer:llama_Entailer, prover_prefix=None, depth=3):
    info = {}
    info["HYP"] = hyp
    info["t5_truth"] = entailer.truthfulness_score(hyp)
    info["llama_truth"] = llama_entailer.truthfulness_score(hyp)
    if depth==0:
        info["t5_faith"] = str(0.0)
        info["llama_faith"] = str(0.0)
        info["premises"] = []
    else:
        info["premises"] = []
        if prover_prefix is not None:
            proof_prefix = "[PREMISE] "+prover_prefix
            proof = entailer.prover({"hypothesis": hyp},  options={"output_prefix": {"proof": proof_prefix}})
        else:
            proof = entailer.prover({"hypothesis": hyp})
        premises = [x.strip() for x in proof.split("[PREMISE]") if x.strip()]

        for premise in premises:
            premise_info = get_score_tree(premise, entailer, llama_entailer, prover_prefix=None, depth=depth-1)
            info["premises"].append(premise_info)
    
    return info
            
    

truth_device = 'cuda:11'
info_device =  'cuda:12'

info_judge = AutoModelForCausalLM.from_pretrained("allenai/truthfulqa-info-judge-llama2-7B").to(info_device)
info_tokenizer = AutoTokenizer.from_pretrained("allenai/truthfulqa-info-judge-llama2-7B", max_length=500)

truth_judge = AutoModelForCausalLM.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B").to(truth_device)
truth_tokenizer = AutoTokenizer.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B", max_length=500)

ew_model = MultiAngleModel(model_path="allenai/entailer-11b", cuda_devices=[13, 14])
prover = NlpAgent(model=ew_model, default_outputs="proof")
entail_verifier = NlpAgent(model=ew_model, default_outputs=["implied"], default_options={"explicit_outputs": ['true', 'false']})
hyp_verifier = NlpAgent(model=ew_model, default_outputs=["valid"], default_options={"explicit_outputs": ['true', 'false']})

entailer = Entailer(ew_model, prover, entail_verifier, hyp_verifier)
llama_entailer = llama_Entailer(ew_model, prover, entail_verifier, truth_judge, truth_tokenizer, info_judge, info_tokenizer, truth_device, info_device)

# with open('fulldepth_obqa.jsonl', 'r') as f:
#     lines = f.readlines()

# with open('fulldepth_obqa_scores.jsonl', 'a') as f:
#     for i in tqdm(range(len(lines))):
#         line = eval(lines[i])
#         result = {}
#         hyp = line["HYP"]
#         result["HYP"] = hyp
#         result["t5_truth"] = str(entailer.truthfulness_score(hyp))
#         result["llama_truth"] = str(llama_entailer.truthfulness_score(hyp))
#         result["t5_faith"] = str(entailer.faithfulness_score(hyp, line["tree"].keys()))
#         result["llama_faith"] = str(llama_entailer.faithfulness_score(hyp, line["tree"].keys()))
#         result["premises"] = get_scores(line["tree"], entailer, llama_entailer)
#         json.dump(result, f)
#         f.write('\n')

# print("scoring quartz trees")

# dataset = load_dataset("allenai/quartz")