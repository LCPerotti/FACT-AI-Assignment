# Adapted from HuggingFace's example code for calculating perplexity
# https://huggingface.co/docs/transformers/en/perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd

import sys
import os
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')
from Src.model import ModelFactory
from Src.experiment import Ablator

class PerplexityExperiment:
    def __init__(self, ablator, base, encodings, batch_size=32, device="cuda"):
        self.ablator = ablator
        self.base = base
        self.encodings = encodings
        self.batch_size = batch_size
        self.device = device

    def calculate_perplexity(self):
        max_length = self.base.config.n_positions
        stride = 512
        seq_len = self.encodings.input_ids.size(1)

        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = self.encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                loss = self.ablator.__run_with_hooks__(input_ids, )

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = loss

            # Accumulate the total negative log-likelihood and the total number of tokens
            num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
        ppl = torch.exp(avg_nll)
        
        return ppl
    
def run(model="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model)
    base = AutoModelForCausalLM.from_pretrained(model)
    test = load_dataset("../data", data_files="wikitext-test.parquet")['train']
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        hooked_model = ModelFactory.create(model)
        ablator = Ablator(model=hooked_model, dataset=test, batch_size=20, experiment="copyVSfact", eval=True)
        if len(head_list[0]) != 0:
            ablator.set_heads(heads=head_list[0], value=boosting_value, position="all")
        if len(head_list[1]) != 0:
            ablator.set_heads(heads=head_list[1], value=suppression_value, position="all")
        
        ppl = PerplexityExperiment(ablator, base, encodings, device=device).calculate_perplexity()
        print(experiment, ppl.item())
        results_buffer.append(ppl.item())
    
    experiments_str = {k: [str(v[0]), str(v[1])] for k, v in experiments.items()}
    df = pd.DataFrame.from_dict(experiments_str, orient='index')

    df.columns = ['boosted_heads', 'suppressed_heads']
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'experiment'}, inplace=True)
    df['perplexity'] = results_buffer

    return df

df = run()
df.to_csv("../results/modification_perplexity.csv", index=False)