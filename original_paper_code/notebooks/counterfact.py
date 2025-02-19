from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
import pandas as pd
import math

import sys
import os
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')
from Src.model import ModelFactory
from Src.experiment import Ablator
from Src.base_experiment import BaseExperiment, to_logit_token

class CounterfactAccuracyExperiment:
    def __init__(self, model_name, dataset_name, device):
        ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
        self.ds = ds.filter(lambda x: x["target_true"].strip() not in x["subject"].strip())
        self.device = device

    def one_token(self, token: torch.Tensor) -> torch.Tensor:
        if token.shape[0] == 1:
            return token
        else:
            return token[0].unsqueeze(0)

    def run(self):
        experiments = {"no_modification": [[], []],
                   #(layer, head)
                   "best_suppression": [[], [(7,10), (9,9), (9,6), (10,0)]], 
                   "best_boost": [[(10,7), (11,10)], []],
                   "best_combined": [[(10,7), (11,10)], [(7,10), (9,9), (9,6), (10,0)]],
                }
        mem_buffer = []
        cp_buffer = []
        other_buffer = []
        
        boosting_value = 5
        suppression_value = 0
        for exp, head_list in experiments.items():
            mem_list = []
            cp_list = []
            other_list = []

            hooked_model = ModelFactory.create("gpt2")
            ablator = Ablator(model=hooked_model, dataset=self.ds, batch_size=20, experiment="copyVSfact", eval=False)
            if len(head_list[0]) != 0:
                ablator.set_heads(heads=head_list[0], value=boosting_value, position="all")
            if len(head_list[1]) != 0:
                ablator.set_heads(heads=head_list[1], value=suppression_value, position="all", reset=False)

            for i in tqdm(range(len(self.ds))):
                item = self.ds[i]
                logit = ablator.__run_with_hooks__(item)
                target_new_token = self.one_token(
                    hooked_model.tokenize(item["target_false"]).squeeze(0).to(self.device)
                )
                target_true_token = self.one_token(
                    hooked_model.tokenize(item["target_true"]).squeeze(0).to(self.device)
                )
                targets = torch.cat(
                    (target_true_token, target_new_token), dim=0
                )

                targets = targets[None, :]
                logit_mem, logit_cp, _, _, mem_winner, cp_winner = to_logit_token(
                    logit, 
                    targets, 
                    normalize="none", 
                    return_winners=True
                )
                
                mem_list.append(mem_winner.item())
                cp_list.append(cp_winner.item())
                other_list.append(1 if cp_winner.item() == 0 and mem_winner.item() == 0 else 0)

            mem_buffer.append(sum(mem_list) / len(mem_list))
            cp_buffer.append(sum(cp_list) / len(cp_list))
            other_buffer.append(sum(other_list) / len(other_list))

        experiments_str = {k: [str(v[0]), str(v[1])] for k, v in experiments.items()}
        df = pd.DataFrame.from_dict(experiments_str, orient='index')

        df.columns = ['boosted_heads', 'suppressed_heads']
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'experiment'}, inplace=True)
        df['mem_rate'] = mem_buffer
        df['cp_rate'] = cp_buffer
        df['other_rate'] = other_buffer

        return df
                
c = CounterfactAccuracyExperiment("gpt2", "counterfact-tracing", "cpu")
df = c.run()
df.to_csv("counterfact_accuracy.csv")
# load dataset
# ablate/boost models (at all token position or just on the subject position)
    # set_heads
# run one prediction per prompt (next token is the one that we care about)
    # run_with_hooks
# we compare this token to the factual token
# aggregate and compute accuracy

        