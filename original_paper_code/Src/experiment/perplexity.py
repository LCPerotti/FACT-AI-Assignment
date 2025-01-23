from transformers import AutoModelForCausalLM, GPT2TokenizerFast
from accelerate.test_utils.testing import get_backend
import torch
from tqdm import tqdm
from datasets import load_dataset

device = "cpu" # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

class PerplexityExperiment:
    def __init__(self, model, encodings, batch_size=16, device="cuda"):
        self.model = model
        self.model.to(device)
        self.encodings = encodings
        self.batch_size = batch_size
        self.device = device

    def reset(self, model, batch_size=16, device="cuda"):
        self.model = model
        self.model.to(device)
        self.batch_size = batch_size
        self.device = device

    def calculate_perplexity(self, hooked=False):
        max_length = self.model.config.n_positions
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
                if hooked:
                    outputs = self.__run_with_hooks__(input_ids)
                else:
                    outputs = self.model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

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
model = AutoModelForCausalLM.from_pretrained("gpt2")
test = load_dataset("../../data", data_files="wikitext-test.parquet")['train']
encodings = tokenizer("\n\n".join(test["text"][7:19]), return_tensors="pt")

ppl = PerplexityExperiment(model, encodings, device="cpu").calculate_perplexity()
print(f"Perplexity: {ppl.item()}")