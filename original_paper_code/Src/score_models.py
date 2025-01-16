import sys
import os  # noqa: F401
import json  # noqa:  F811
import pandas as pd

import torch  # noqa: F401
from torch.utils.data import DataLoader  # noqa: F401
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from tqdm import tqdm  # noqa: F401
from typing import Literal, Optional, Tuple  # noqa: F401
from dataclasses import dataclass  # noqa: F401


sys.path.append("..")
sys.path.append("../src")
sys.path.append("../data")
from Src.dataset import BaseDataset  # noqa: E402
from Src.model import BaseModel  # noqa: E402

FILENAME = "../results/{}_evaluate_mechanism_data_sampling.csv"

class EvaluateMechanism:
    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        device="cpu",
        batch_size=100,
        similarity: Tuple[
            bool, int, Literal["data-sampling", "modify-self-similarity", "self-similarity", "ungrouped"]
        ] = (True, 4, "self-similarity"),
        premise="Redefine",
        family_name: Optional[str] = None,
        num_samples=1,
    ):
        self.tokenizer = model.get_tokenizer()
        self.model = model
        self.model = self.model.to(device)
        self.model_name = model.cfg.model_name
        self.dataset = dataset
        # self.lenghts = self.dataset.get_lengths()
        self.device = device
        self.batch_size = batch_size
        self.similarity = similarity
        self.premise = premise
        self.family_name = family_name
        self.n_samples = num_samples
        print("Model device", self.model.device)
        print("sim catgory", self.similarity)

    def update(
        self,
        dataset: BaseDataset,
        similarity: Tuple[bool, int, Literal["self-similarity", "modify-self-similarity","data-sampling", "ungrouped"]],
        premise: str,
    ):
        self.dataset = dataset
        self.similarity = similarity
        self.premise = premise

    def check_prediction(self, logits, target):
        """Checks whether a model predicts the factual or counterfactual
        target, or another result. Returns BATCH indices of the samples that
        were predicted as target_true (F), target_false (CF), or other."""
        #print("Logit shape",logit.shape) 
        # apply softmax and get probability of the last token at the end of sequence
        probs = torch.softmax(logits, dim=-1)[:, -1, :]
        #print("Probs shape", probs.shape)
        # count the number of times the model predicts the target[:, 0] or target[:, 1]
        num_samples = target.shape[0]
        target_true = 0
        target_false = 0
        other = 0
        target_true_indices = []
        target_false_indices = []
        other_indices = []
        for i in range(num_samples):
            max_prob = torch.argmax(probs[i])  #TODO is this the correct index?
            #print("Max prob", self.model.tokenizer.decode(max_prob), "Target true", self.model.tokenizer.decode(target[i,0]), "Target false", self.model.tokenizer.decode(target[i,1]))
            if max_prob == target[i, 0]:  #TODO is this the correct index?
                # print("DEBUG:", self.tokenizer.decode(target[i, 0]), self.tokenizer.decode(torch.argmax(probs[i])))
                target_true += 1
                target_true_indices.append(i)
            elif max_prob == target[i, 1]:
                target_false += 1
                target_false_indices.append(i)
            else:
                other += 1
                other_indices.append(i)
        return target_true_indices, target_false_indices, other_indices

    def evaluate(self, dataloader):

        target_true, target_false, other = 0, 0, 0
        n_batch = len(dataloader)
        all_true_indices = []
        all_false_indices = []
        all_other_indices = []

        idx = 0
        for batch in tqdm(dataloader, desc="Extracting logits"):
            #print(len(batch))                
            input_ids = batch["input_ids"].to(self.device)
            #print("Input ids shape", input_ids.shape)
            output = self.model(input_ids)
            #print("Output type", type(output))
            logits = output["logits"]
            # vocab_items = self.tokenizer.batch_decode( batch["target"])
            # print("Target", vocab_items, batch["target"].shape)
            count = self.check_prediction(logits, batch["target"])
            #print("The count produced is", count)
            target_true += len(count[0])
            target_false += len(count[1])
            other += len(count[2])
            #print("Valid sum?", len(count[0]) + len(count[1]) + len(count[2]), input_ids.shape[0])

            all_true_indices.extend(
                [
                    self.dataset.original_index[i + idx * self.batch_size]
                    for i in count[0]
                ]
            )
            all_false_indices.extend(
                [
                    self.dataset.original_index[i + idx * self.batch_size]
                    for i in count[1]
                ]
            )
            all_other_indices.extend(
                [
                    self.dataset.original_index[i + idx * self.batch_size]
                    for i in count[2]
                ]
            )
            idx += 1
        return (
            target_true,
            target_false,
            other,
            all_true_indices,
            all_false_indices,
            all_other_indices,
        )

    def evaluate_all(self):
        """What on earth is this function supposed to do?"""
        target_true, target_false, other = [], [], []
        #print("SAMPLES", self.n_samples)
        
        #self.dataset.reset(new_similarity_level = self.similarity[1]) # this and the previous line seem messy
        print("Length of dataset", len(self.dataset.full_data))
        target_true_tmp, target_false_tmp, other_tmp = 0, 0, 0
        all_true_indices = []
        all_false_indices = []
        all_other_indices = []
        for length in self.dataset.get_lengths():
            self.dataset.set_len(length)
            if len(self.dataset) == 0:
                continue
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
            result = self.evaluate(dataloader)
            # target_true_tmp += result[0]
            # target_false_tmp += result[1]
            # other_tmp += result[2]

            # assert duplicates in the indices
            assert set(result[3]) & set(result[4]) & set(result[5]) == set(), "Duplicates in the indices"

            all_true_indices.extend(result[3])
            all_false_indices.extend(result[4])
            all_other_indices.extend(result[5])

        for i in all_true_indices:
            self.dataset.full_data[i]["parse"] = "true"
        for i in all_false_indices:
            self.dataset.full_data[i]["parse"] = "false"
        for i in all_other_indices:
            self.dataset.full_data[i]["parse"] = "other"

        if len(self.model_name.split("/")) > 1:
            save_name = self.model_name.split("/")[1]
        else:
            save_name = self.model_name
        if self.similarity[0]:
            save_name += "similarity"

        self.dataset.as_dataframe().to_csv(f"../results/sim_raw_results_{save_name}.tsv", sep="\t", index=True)
        print("Saved results!")
        return


        # # add the results of the sample to the total
        # target_true.append(target_true_tmp)
        # target_false.append(target_false_tmp)
        # other.append(other_tmp)

        # #print("Target true length", len(target_true))
        # # average the results over the number of samples
        # Target_true = target_true
        # Target_false = target_false
        # Other = other
        # print("target true", Target_true)
        # print("target false", Target_false)
        # print("other", Other)

        # target_true = torch.mean(torch.tensor(Target_true).float())
        # target_false = torch.mean(torch.tensor(Target_false).float())
        # other = torch.mean(torch.tensor(Other).float())

        # target_true_std = torch.std(torch.tensor(Target_true).float())
        # target_false_std = torch.std(torch.tensor(Target_false).float())
        # other_std = torch.std(torch.tensor(Other).float())

        # print(
        #     f"Total: Target True: {target_true}, Target False: {target_false}, Other: {other}"
        # )
        # print(
        #     f"Total: Target True std: {target_true_std}, Target False std: {target_false_std}, Other std: {other_std}"
        # )

        # index = torch.cat(index, dim=1)

        if len(self.model_name.split("/")) > 1:
            save_name = self.model_name.split("/")[1]
        else:
            save_name = self.model_name
        if self.similarity[0]:
            save_name += "similarity"
        # save results

        filename = FILENAME.format(self.family_name)
        # if file not exists, create it and write the header
        if not os.path.isfile(filename):
            with open(filename, "w") as file:
                file.write(
                    "model_name,orthogonalize,premise,interval,similarity_type,target_true,target_false,other,target_true_std,target_false_std,other_std\n"
                )

        with open(filename, "a+") as file:
            file.seek(0)
            # if there is aleardy a line with the same model_name and orthogonalize, delete it
            lines = file.readlines()
            # Check if a line with the same model_name and orthogonalize exists
            line_exists = any(
                line.split(",")[0] == self.model_name
                and line.split(",")[1] == str(self.similarity[0])
                and line.split(",")[2] == self.premise
                and line.split(",")[3] == self.similarity[1]
                and line.split(",")[4] == self.similarity[2]
                for line in lines
            )

            # If the line exists, remove it
            if line_exists:
                lines = [
                    line
                    for line in lines
                    if not (
                        line.split(",")[0] == self.model_name
                        and line.split(",")[1]
                        == str(
                            self.similarity[0]
                            and line.split(",")[2] == self.premise
                            and line.split(",")[3] == self.similarity[1]
                            and line.split(",")[4] == self.similarity[2]
                        )
                    )
                ]

                # Rewrite the file without the removed line
                file.seek(0)  # Move the file pointer to the start of the file
                file.truncate()  # Truncate the file (i.e., remove all content)
                file.writelines(lines)  # Write the updated lines back to the file
            file.write(
                f"{self.model_name},{self.similarity[0]},{self.premise},{self.similarity[1]},{self.similarity[2]} ,{target_true},{target_false},{other},{target_true_std},{target_false_std},{other_std}\n"
            )

        # save indices
        if not os.path.isdir(f"../results/{self.family_name}_evaluate_mechs_indices"):
            # if the directory does not exist, create it
            os.makedirs(f"../results/{self.family_name}_evaluate_mechs_indices")

        with open(
            f"../results/{self.family_name}_evaluate_mechs_indices/{save_name}_evaluate_mechanism_indices.json",
            "w",
        ) as file:
            json.dump(
                {
                    "target_true": all_true_indices,  # type: ignore
                    "target_false": all_false_indices,  # type: ignore
                    "other": all_other_indices,  # type: ignore
                },
                file,
            )

        return target_true, target_false, other
