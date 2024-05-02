# Multistep-reasoning

We improve the errors of entailer due to incorrect beliefs of [entailer](https://arxiv.org/abs/2210.12217) by using LLaMa2-7B for scoring the truthfulness of the premise. We change the reasoned scoring of the hypothesis to use the geometric mean of the truthfulness scores of the premises. A short walkthrough of code is available at `walkthrough.ipynb`.

## Entailer and LLaMa Entailer
- Entailer is defined in `entailer.py`. It uses t5-model from [entailer](https://arxiv.org/abs/2210.12217) for generating one-step explanation and scoring truthfulness and faithfulness of premises.
- llama Entailer is defined in `llama_entailer.py`. It uses llama2-7B model for scoring truthfulness and faithfulness of the premises.
- Following are the methods defined in these classes:
    -   `truthfulness_score`: scores the truthfulness of the prompt
    -   `faithfulness_score`: scores the faithfulness of the premises with hypothesis.
    - `one_step`: generates a one step explanation for the given hypothesis.
    - `generate_entailment_tree`: generate an entailment for the hypothesis for a given max depth.

## Generating full-depth score trees
- Since we are imporving errors of entailer due to incorrect beliefs we generate a full-depth score tree for every hypothesis in the dataset to avoid generating the premises again and again.
- `truth_faith_score.py` is used to generate score-tree for the hypothesis. 
- We generate and store the score trees of obqa, quartz and truthfulqa at `results/`.

## Finetuning the LLaMA2-7B model
- Finetuning is available at `truthfulqa_reeval/scripts/finetune_judge.sh`. We modified the code from [`yizhongw/truthfulqa_reeval`](https://github.com/yizhongw/truthfulqa_reeval/tree/main).
- We created dataset using ARC and worldtree datasets it is located at `truthfulqa_reeval/data`.

## Re-evaluating the scores
- We use the finetuned-model to modify the scores in score-tree generated earlier.
- `modify_scores.py` is used to modify the truthfulness and faithfulness scores using a custom model.
- We store the modified score-trees at `results/`

## Ablation Study
- We evaluate the following approaches using the score-trees obtained
    - **Direct**: Choosing the answer corresponding to the highest-scores hypothesis.
    - **Entailer**: Expanding the entailment tree up to a certain
depth, then using only the truthfulness scores at the leaf
noded and the faithfulness scores at all levels backpropagat-
ing the reasoned score.
    - **Entailer+Direct**: In this case, a node is expanded only is the
reasoned score, which is based on the truthfulness scores of
the child premises and the faithfulness score of the entail-
ment is higher than the score of the node itsel
- We observe that using logit-transform and taking the geometric
mean of the truthfulness scores works well.
- We show comparison
across different parameters at `results/results.ods`