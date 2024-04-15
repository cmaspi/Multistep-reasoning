import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from entailment_bank.utils.nlp_agent import MultiAngleModel, NlpAgent


class llama_Entailer:
    def __init__(self, model, prover, entail_verifier, truth_judge, truth_tokenizer, truth_device='cuda:7'):
        self.model = model
        self.prover = prover
        self.entail_verifier = entail_verifier
        self.truth_judge = truth_judge
        self.truth_tokenizer = truth_tokenizer
        self.truth_device = truth_device
        

    def truthfulness_score(self, prompt):
        device = self.truth_device
        prompt = prompt+"\nTrue"
        inputs = self.truth_tokenizer([prompt], return_tensors="pt").to(device)
        outputs = self.truth_judge.generate(**inputs, output_scores=True, return_dict_in_generate=True, max_length=100)
        input_length = 1 if self.truth_judge.config.is_encoder_decoder else inputs.input_ids.shape[1]
        transition_scores = self.truth_judge.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        generated_tokens = outputs.sequences[:,input_length:]
        score = transition_scores[0][0]
        score = np.exp(score.numpy(force=True))
        if self.truth_tokenizer.decode(generated_tokens[0][0])=='yes':
            return score
        return 1.0-score
        
    def one_step(self, hyp, prover_prefix, k=1):
        """
        Finds the best set of premises P which entail the hypothesis H
        """
        premises_dict = {}
        for i in range(k):
            if prover_prefix is not None:
                proof_prefix = "[PREMISE] "+prover_prefix
                proof = self.prover({"hypothesis": hyp},  options={"output_prefix": {"proof": proof_prefix}})
            else:
                proof = self.prover({"hypothesis": hyp})
            premises = [x.strip() for x in proof.split("[PREMISE]") if x.strip()]

            premise_score = [None]*len(premises)

            for j in range(len(premises)):
                premise_score[j] = self.truthfulness_score(premises[j])

            entail_res = self.entail_verifier({"hypothesis": hyp, "proof": proof})
            entail_score = entail_res['output_prob'] if entail_res['implied']=='true' else 1-entail_res['output_prob']

            if min(premise_score)<0.5 or entail_score<0.5:
                continue

            score = entail_score
            for j in range(len(premises)):
                score *= premise_score[j]

            premises_dict[proof] = score

        if len(premises_dict.keys())>0:
            best_premises = max(premises_dict, key=lambda x: premises_dict[x])
        else:
            best_premises = None

        return best_premises
    
    def generate_entailment_tree(self, hyp, visited, max_depth=3, k=5, prover_prefix=None):
        """
        Entailer's backchaining algorithm for searching for the best proof tree(H) and score s(H) for a hypothesis H.
        """
        tree = {}
        
        visited[hyp] = True
        
        sd_H = self.truthfulness_score(hyp)
        cd_H = max(sd_H, 1-sd_H)
        
        if max_depth == 0:
            return sd_H, {}, {}, 0

        P = self.one_step(hyp, prover_prefix, k=k)

        if P is None:
            return sd_H,{},{},0

        premises = [x.strip() for x in P.split("[PREMISE]") if x.strip()]

        ent_res = self.entail_verifier({"hypothesis": hyp, "proof":P})
        entail_score = ent_res['output_prob'] if ent_res['implied']=='true' else 1-ent_res['output_prob']
        
        p_tree = [None]*len(premises)
        p_score = [None]*len(premises)
        
        prem_scores = {}

        if entail_score>cd_H:
            premises_score = 1
            c = 0
            for i in range(len(premises)):
                if premises[i] not in visited.keys():
                    p_score[i], p_tree[i], _, _ = self.generate_entailment_tree(premises[i], visited, max_depth-1, k, prover_prefix)
                    prem_scores[premises[i]] = p_score[i]
                    premises_score*=p_score[i]
                    c+=1
            
            sr_H = entail_score
            if c>0:
                sr_H = sr_H*(premises_score**(1/c))
                    
        else:
            for i in range(len(premises)):
                p_score[i] = self.truthfulness_score(premises[i])
                prem_scores[premises[i]] = p_score[i]
            sr_H = 0

        cr_H = sr_H

        # if reasoning confidence is higher we expand the node
        if cr_H > cd_H:
            tree_score = sr_H
            for i in range(len(premises)):
                tree[premises[i]] = p_tree[i]
        else:
            tree_score = sd_H

        return tree_score, tree, prem_scores, entail_score
    
if __name__=="__main__":
    device = 'cuda:7'
    truth_judge = AutoModelForCausalLM.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B")
    tokenizer = AutoTokenizer.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B", max_length=100)
    
    hyp = "Predators eat prey"
    
    print("HYP: ", hyp)
    ew_model = MultiAngleModel(model_path="allenai/entailer-11b", cuda_devices=[10, 11])
    prover = NlpAgent(model=ew_model, default_outputs="proof")
    entail_verifier = NlpAgent(model=ew_model, default_outputs=["implied"], default_options={"explicit_outputs": ['true', 'false']})

    entailer = llama_Entailer(ew_model, prover, entail_verifier, truth_judge, tokenizer)
    tree_score, tree, _, _ = entailer.generate_entailment_tree(hyp, {}, 1, 1)
    print("Tree: ", tree)
    