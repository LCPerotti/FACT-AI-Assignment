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
        results_buffer = []
        boosting_value = 5
        suppression_value = 0
        for experiment, head_list in experiments.items():
            hooked_model = ModelFactory.create("gpt2")
            ablator = Ablator(model=hooked_model, dataset=self.ds, batch_size=20, experiment="copyVSfact", eval=False)
            if len(head_list[0]) != 0:
                ablator.set_heads(heads=head_list[0], value=boosting_value, position="all")
            if len(head_list[1]) != 0:
                ablator.set_heads(heads=head_list[1], value=suppression_value, position="all")

            for i in tqdm(range(0, len(self.ds))):
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
                print(target_true_token, target_new_token, targets)
                print(logit.shape, targets.shape)
                logit_mem, logit_cp, _, _, mem_winner, cp_winner = to_logit_token(
                    logit, 
                    targets, 
                    normalize="none", 
                    return_winners=True
            )
                print(mem_winner, cp_winner)
                
                # mem_winners, cp_winners = ablator.get_winners(logit_mem, logit_cp)
                # results_buffer.append(mem_winners == cp_winners)

c = CounterfactAccuracyExperiment("gpt2", "counterfact-tracing", "cpu")
c.run()
# load dataset
# ablate/boost models (at all token position or just on the subject position)
    # set_heads
# run one prediction per prompt (next token is the one that we care about)
    # run_with_hooks
# we compare this token to the factual token
# aggregate and compute accuracy

        