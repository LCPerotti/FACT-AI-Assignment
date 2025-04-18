# Changes made to run original code
Using the original code and `requirements.txt` file, we were not able to get the code running.
As such, we made a different environment file and made changes in various files to be able to run the original code, with the intent to leave the results unaltered, but to be able to run the original experiments without errors.
These changes are recorded below, per file.

## environment.yml (new)
Created since the `requirements.txt` do not include all needed modules, and the modules required a specific python version.

- Added missing pytest dependency
- Updates required tqdm version (4.66.1 -> 4.66.3)
- Set specific scipy version for use of deprecated functions used (scipy <= 1.12)
- Removed exact version for transformerlens and transformers to allow more recent versions and automated version resolution by pip

## Script/run_all.py
Threw an error when attempted to run due to missing values and undeclared variables.

- Removed references to missing `Src.config` module, set `hf_access_token` and `hf_model_cache_dir` to empty.
These appear to be only used for the Llama 2 model, which we did not use.
- Set `config.data_slice` to `None`, since otherwise the value is not set. This value is only used for determining file names.

## Src/dataset.py
Threw an error when attempted to run due to missing import.

- Removed `line_reader` module import, since it is unneeded for running the experiments and not included in the `requirements.txt`

## Src/experiment/logit_attribution.py
Threw an error when attempted to run due to undeclared variables.

- Uncommented an existing line adding `apply_ln` with default value `False` as an argument for `run`, since otherwise the value is not passed.
