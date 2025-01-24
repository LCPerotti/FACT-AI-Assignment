from transformers import AutoModelForCausalLM, GPT2TokenizerFast
from accelerate.test_utils.testing import get_backend
import torch
from tqdm import tqdm
from datasets import load_dataset

import sys
import os
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')
from Src.model import ModelFactory
from Src.experiment import Ablator

device = "cpu" # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

class PerplexityExperiment:
    def __init__(self, lm_interface, base, encodings, batch_size=16, device="cuda", hooked=False):
        self.lm_interface = lm_interface
        self.base = base
        if not hooked:
            self.lm_interface.to(device)
        self.encodings = encodings
        self.batch_size = batch_size
        self.device = device
        self.hooked = hooked

    def calculate_perplexity(self):
        max_length = self.base.config.n_positions
        stride = 16
        seq_len = self.encodings.input_ids.size(1)

        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = self.encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                if self.hooked:
                    outputs = self.lm_interface.__run_with_hooks__(input_ids, )
                else:
                    outputs = self.lm_interface(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss if not self.hooked else outputs

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


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
base = AutoModelForCausalLM.from_pretrained("gpt2")
hooked_model = ModelFactory.create("gpt2")
ablator = Ablator(model=hooked_model, dataset=[], batch_size=20, experiment="copyVSfact", eval=True)
ablator.set_heads(heads=[(10,7), (11,10), (11,1), (11,3)], value=0, position="all")
test = load_dataset("../data", data_files="wikitext-test.parquet")['train']
encodings = tokenizer("\n\n".join(test["text"][:20]), return_tensors="pt")

ppl = PerplexityExperiment(ablator, base, encodings, device="cpu", hooked=True).calculate_perplexity()
print(f"Perplexity: {ppl.item()}")
ppl2 = PerplexityExperiment(base, base, encodings, device="cpu", hooked=False).calculate_perplexity()
print(f"Perplexity base: {ppl2.item()}")
print("diff", ppl2.item() - ppl.item())