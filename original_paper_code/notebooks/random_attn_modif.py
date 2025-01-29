import sys
import os
import pandas as pd
import numpy as np
import itertools
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

from Src.model import ModelFactory
model = ModelFactory.create("gpt2")

from Src.dataset import BaseDataset
from Src.experiment import Ablator


dataset = BaseDataset(path = "../data/full_data_sampled_gpt2.json",
                      model = model,
                      experiment="copyVSfact",
                      no_subject=True)
ablator = Ablator(model=model, dataset=dataset, experiment="copyVSfact", batch_size=20)

results = []

tried_heads = []
while len(tried_heads) < 20:
    heads = np.random.randint(0, 12), np.random.randint(0, 12)
    if heads not in tried_heads:
        tried_heads.append(heads)
        ablator.set_heads([heads], position="attribute", value=5, reset=True)
        df = ablator.run()
        df['heads'] = str(heads)
        df['experiment'] = 'single_boost'
        results.append(df)

results = pd.concat(results)
results.to_csv("single_boost3.csv", index=False)

results2 = []
tried_tuples = []

while len(tried_tuples) < 20:
    head1, head2 = sorted(((np.random.randint(0, 12), np.random.randint(0, 12)) for _ in range(2)))
    print(head1, head2)
    if (head1, head2) not in tried_tuples:
        tried_tuples.append((head1, head2))
        ablator.set_heads([head1, head2], position="attribute", value=5, reset=True)
        df = ablator.run()
        df['heads'] = f'{head1}, {head2}'
        df['experiment'] = 'double_boost'
        df['layer_diff'] = abs(head1[0] - head2[0])
        print(df)
        results2.append(df)

results2 = pd.concat(results2)
results2.to_csv("double_boost3.csv", index=False)