# Standard library imports
from json import load
import json
import sys
import os


sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("../Src")))
sys.path.append(os.path.abspath(os.path.join("../data")))
from dataclasses import dataclass

# NOTE: Disabled values from Src.config for now, to prevent input error
# This probably results in the models being cached in Script folder, which seems fine for now.
# from Src.config import hf_access_token, hf_model_cache_dir # noqa: E402
hf_access_token = ""
hf_model_cache_dir = ""
os.environ["HF_HOME"] = hf_model_cache_dir
from re import A
import io
import subprocess
from typing import Optional, Literal, Union


# Third-party library imports
from rich.console import Console
import argparse
import logging
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
import ipdb
# Local application/library specific imports

from Src.dataset import BaseDataset  # noqa: E402
from Src.experiment import LogitAttribution, LogitLens,  Ablate, HeadPattern  # noqa: E402
from Src.model import BaseModel, ModelFactory  # noqa: E402
from Src.utils import display_config, display_experiments, check_dataset_and_sample  # noqa: E402
console = Console()
# set logging level to suppress warnings
logging.basicConfig(level=logging.ERROR)



def get_hf_model_name(model_name):
    """
    Get the string for the location of the model on HuggingFace
    """
    if "Llama" in model_name:
        return "meta-llama/" + model_name
    elif "opt" in model_name:
        return "facebook/" + model_name
    elif "pythia" in model_name:
        return "EleutherAI/" + model_name
    elif "gpt2" in model_name:
        return model_name
    else:
        raise ValueError("No HF model name found for model name: ", model_name)




@dataclass
class Config:
    """
    Stores the configurations of the experiment to be run from the input args.
    Stores a copy of the config in /results/.
    """
    experiment: Literal["copyVSfact"] = "copyVSfact"
    model_name: str = "gpt2"
    hf_model_name: str = "gpt2"
    hf_model: bool= False
    batch_size: int = 10
    dataset_path: str = f"../data/full_data_sampled_{model_name}_with_subject.json"
    dataset_start: Optional[int] = None
    dataset_end: Optional[int] = None
    produce_plots: bool = True
    normalize_logit: Literal["none", "softmax", "log_softmax"] = "none"
    std_dev: int = 1  # 0 False, 1 True
    total_effect: bool = False
    up_to_layer: Union[int, str] = "all"
    ablate_component:str = "all"
    flag: str = ""

    @classmethod
    def from_args(cls, args):
        """
        Reads most values for the configuration from the args provided in the command line.
        Reads the following values:
        - type of experiment
        - model_name
        - batch size
        - whether to produce plots
        - std-dev
        - total-effect
        - start and end of dataset part
        - ablate_component
        - dataset start and end.
        - flag
        """
        return cls(
            experiment=args.experiment,
            model_name=args.model_name,
            batch_size=args.batch,
            dataset_path= get_dataset_path(args),
            dataset_start=args.start,
            dataset_end=args.end,
            produce_plots=args.produce_plots,
            std_dev=1 if not args.std_dev else 0,
            total_effect=args.total_effect if args.total_effect else False,
            hf_model_name= get_hf_model_name(args.model_name),
            ablate_component=args.ablate_component,
            flag = args.flag,
            normalize_logit = args.normalize_logit
        )
        
    def to_json(self):
        return {
            "experiment": self.experiment,
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "dataset_path": self.dataset_path,
            "dataset_start": self.dataset_start,
            "dataset_end": self.dataset_end,
            "produce_plots": self.produce_plots,
            "std_dev": self.std_dev,
            "total_effect": self.total_effect,
            "hf_model_name": self.hf_model_name,
            "ablate_component": self.ablate_component,
            "flag": self.flag,
            "normalize_logit": self.normalize_logit

        }

# Gets the dataset needed for the experiment.

def get_dataset_path(args):
    """
    Gets the path for the dataset based on the type of experiment.
    Only supports the experiment "copyVSfact"
    """
    if args.experiment == "copyVSfact":
        return f"../data/full_data_sampled_{args.model_name}_with_subjects.json"
    else:
        raise ValueError("No dataset path found for folder: ", args.folder)

@dataclass
class logit_attribution_config:
    std_dev: int = 0  # 0 False, 1 True


@dataclass
class logit_lens_config:
    component: str = "resid_post"
    return_index: bool = True


### check folder and create if not exists
def save_dataframe(folder_path, file_name, dataframe):
    """
    Checks whether the folder exists for the dataframe, and if not, creates it.
    Afterwards, stores the dataframe in there as a csv.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    dataframe.to_csv(f"{folder_path}/{file_name}.csv", index=False)


def logit_attribution(model, dataset, config, args):
    """
    Runs the logit attribution experiment (mainly defined in experiment/logit_attribution.py).
    Also decides the name for where to store the data. Which is:
    /results/<experimentType><flag>/logit_attribution/<model_name>_<dataset_slice_name>
    Note: experimentType will likely always be "copyVSfact" since it is the only accepted one.
    If --no-plots isn't set, also creates a plot using "src_figure/logit_attribution.R"/
    """

    dataset_slice_name = "full" if config.dataset_end is None else config.dataset_end
    dataset_slice_name = (
        dataset_slice_name if config.up_to_layer == "all" else f"{dataset_slice_name}_layer_{config.up_to_layer}"
    )
    print("Running logit attribution")
    attributor = LogitAttribution(dataset, model, config.batch_size // 5, config.experiment)
    dataframe = attributor.run(apply_ln=False, normalize_logit=config.normalize_logit, up_to_layer=config.up_to_layer)
    save_dataframe(
        f"../results/{config.experiment}{config.flag}/logit_attribution/{config.model_name}_{dataset_slice_name}",
        "logit_attribution_data",
        dataframe,
    )

    if config.produce_plots:
        # run the R script
        logit_attribution_plot(config, dataset_slice_name)

def logit_attribution_plot(config, dataset_slice_name):
        """
        Creates a subprocess to run the R script in "../src_figure/logit_attribution.R".
        This seemingly creates the logit_attribution plot.
        Reads the data from folder: /results/<experimentType><flag>/logit_attribution/<model_name>_<dataset_slice_name>
        """
        subprocess.run(
            [
                "Rscript",
                "../src_figure/logit_attribution.R",
                f"../results/{config.experiment}{config.flag}/logit_attribution/{config.model_name}_{dataset_slice_name}",
                f"{config.std_dev}",
            ]
        )


def logit_lens(model, dataset, config, args):
    """
    Runs the logit attribution experiment (mainly defined in experiment/logit_lens.py).
    Also decides the name for where to store the data. Which is:
    /results/<experimentType><flag>/logit_lens/<model_name>_<dataset_slice_name>
    Note: experimentType will likely always be "copyVSfact" since it is the only accepted one.
    If --no-plots isn't set, also creates a plot using "src_figure/logit_lens.R"/
    """
    data_slice_name = "full" if config.dataset_end is None else config.dataset_end
    logit_lens_cnfg = logit_lens_config()
    print("Running logit lens")
    logit_lens = LogitLens(dataset, model, config.batch_size, config.experiment)
    dataframe = logit_lens.run(
        logit_lens_cnfg.component,
        logit_lens_cnfg.return_index,
        normalize_logit=config.normalize_logit,
    )
    save_dataframe(
        f"../results/{config.experiment}{config.flag}/logit_lens/{config.model_name}_{data_slice_name}",
        "logit_lens_data",
        dataframe,
    )

    if config.produce_plots:
        # run the R script
        logit_lens_plot(config, data_slice_name)
        
def logit_lens_plot(config, data_slice_name):
        """
        Creates a subprocess to run the R script in "../src_figure/logit_attribution.R".
        This seemingly creates the logit_attribution plot.
        Reads the data from folder: /results/<experimentType><flag>/logit_lens/<model_name>_<dataset_slice_name>
        """
        print("Plotting from source:", f"../results/{config.experiment}/logit_lens/{config.model_name}_{data_slice_name}")
        subprocess.run(
            [
                "Rscript",
                "../src_figure/logit_lens.R",
                f"../results/{config.experiment}{config.flag}/logit_lens/{config.model_name}_{data_slice_name}",
            ]
        )


def ablate(model, dataset, config, args):
    """
    Runs the attention ablation/modification experiment (mainly defined in experiment/ablator.py).
    Also decides the name for where to store the data. Which is:
    /results/<experimentType><flag>/ablation/<model_name>_<dataset_slice_name>
    Note: experimentType will likely always be "copyVSfact" since it is the only accepted one.
    If --no-plots isn't set, also creates a plot using the ablate_plot function.
    """
    data_slice_name = "full" if config.dataset_end is None else config.dataset_end
    start_slice_name = "" if config.dataset_start is None else f"{config.dataset_start}_"
    data_slice_name = f"{start_slice_name}{data_slice_name}_total_effect" if config.total_effect else data_slice_name
    LOAD_FROM_PT = None
    ablator = Ablate(dataset, model, config.batch_size, config.experiment)
    if args.ablate_component == "all":
        dataframe, tuple_results = ablator.run_all(normalize_logit=config.normalize_logit, total_effect=args.total_effect, load_from_pt=LOAD_FROM_PT)
        save_dataframe(
            f"../results/{config.experiment}{config.flag}/ablation/{config.model_name}_{data_slice_name}",
            "ablation_data",
            dataframe,
        )
        torch.save(tuple_results, f"../results/{config.experiment}{config.flag}/ablation/{config.model_name}_{data_slice_name}/ablation_data.pt")
    else:
        dataframe, tuple_results = ablator.run(args.ablate_component, normalize_logit=config.normalize_logit, total_effect=args.total_effect, load_from_pt=LOAD_FROM_PT)
        save_dataframe(
            f"../results/{config.experiment}{config.flag}/ablation/{config.model_name}_{data_slice_name}",
            f"ablation_data_{args.ablate_component}",
            dataframe,
        )
        torch.save(dataframe, f"../results/{config.experiment}{config.flag}/ablation/{config.model_name}_{data_slice_name}/ablation_data_{args.ablate_component}.pt")

    if config.produce_plots:
        # run the R script
        ablate_plot(config, data_slice_name)
        
def ablate_plot(config, data_slice_name):
    """
    Creates a subprocess to run the R script in "../src_figure/ablation.R".
    This seemingly creates the ablation plot.
    Reads the data from folder: /results/<experimentType><flag>/ablation/<model_name>_<dataset_slice_name>,
    """

    data_slice_name = f"{data_slice_name}_total_effect" if config.total_effect else data_slice_name
    print("plotting from source: ",  f"../results/{config.experiment}/ablation/{config.model_name}_{data_slice_name}")
    subprocess.run(
        [
            "Rscript",
            "../src_figure/ablation.R",
            f"../results/{config.experiment}{config.flag}/ablation/{config.model_name}_{data_slice_name}",
            f"{config.std_dev}",
        ]
    )
        
def pattern(model, dataset, config, args):
    """
    Runs the attention pattern analysis (mainly defined in experiment/head_pattern.py).
    Also decides the name for where to store the data. Which is:
    /results/<experimentType><flag>/ablation/<model_name>_<dataset_slice_name>
    Note: experimentType will likely always be "copyVSfact" since it is the only accepted one.
    If --no-plots isn't set, also creates a plot using the pattern_plot function.
    """
    data_slice_name = "full" if config.dataset_end is None else config.dataset_end
    print("Running head pattern")
    pattern = HeadPattern(dataset, model, config.batch_size, config.experiment)
    dataframe = pattern.run()
    save_dataframe(
        f"../results/{config.experiment}{config.flag}/head_pattern/{config.model_name}_{data_slice_name}",
        "head_pattern_data",
        dataframe,
    )

    if config.produce_plots:
        # run the R script
        pattern_plot(config, data_slice_name)
        

def pattern_plot(config, data_slice_name):
        """
        Creates a subprocess to run the R script in "../src_figure/head_pattern.R".
        This seemingly creates the attention pattern plot, possibly Figure 10 in the paper.
        Reads the data from folder: /results/<experimentType><flag>/head_pattern/<model_name>_<dataset_slice_name>
        """
        subprocess.run(
            [
                "Rscript",
                "../src_figure/head_pattern.R",
                f"../results/{config.experiment}{config.flag}/head_pattern/{config.model_name}_{data_slice_name}",
            ]
        )

class CustomOutputStream(io.StringIO):
    """
    A class for an output stream that seems specialized for displaying experiments.
    Does not appear to be used anywhere else in the code, as far as I can see.
    """
    def __init__(self, live, index, status, experiments):
        super().__init__()
        self.live = live
        self.index = index
        self.status = status
        self.experiments = experiments

    def write(self, text):
        super().write(text)
        self.status[self.index] = text
        self.live.update(display_experiments(self.experiments, self.status))

def load_model(config) -> BaseModel:
    """
    Creates a model (subclass of BaseModel) from the model name, and whether it is a huggingface model. Also sends the model to a cuda device if available.
    If hf_model is False, the model is wrapped with the WrapHookedTransformer class, otherwise, it is wrapped with the WrapAutoModelForCausalLM class.
    """
    return ModelFactory.create(config.model_name, config.hf_model)

def main(args):
    """
    Called by the __main__ class after parsing.
    Gathers the config from the argparser.
    If --only-plots is set, only runs the plot creation code for any selected experiments, as long as the data for it is present.
    Otherwise, runs all experiments that were selected in the config, and displays
    the status of the experiments.
    """
    config = Config().from_args(args)
    console.print(display_config(config))
    # create experiment folder
    if not os.path.exists(f"../results/{config.experiment}"):
        os.makedirs(f"../results/{config.experiment}")
        
    # save the config in a json file in the experiment folder
    with open(f"../results/{config.experiment}/config.json", "w") as f:
        json.dump(config.to_json(), f)

    # NOTE: Set slice to None to avoid error
    config.dataset_slice = None

    # create experiment folder
    if args.only_plot:
        data_slice_name = "full" if config.dataset_end is None else config.dataset_end

        def try_to_run_plot(plot_function):
            try:
                plot_function(config, data_slice_name)
            except FileNotFoundError:
                print(f"No {plot_function.__name__} data found")
        
        plots = []
        if args.logit_attribution:
            plots.append(logit_attribution_plot)
        if args.logit_lens:
            plots.append(logit_lens_plot)
        if args.ablate:
            plots.append(ablate_plot)
        if args.pattern:
            plots.append(pattern_plot)
        if args.all:
            plots = [logit_attribution_plot, logit_lens_plot,  ablate_plot, pattern_plot]
            
        for plot in plots:
            try_to_run_plot(plot)
        return

    check_dataset_and_sample(config.dataset_path, config.model_name, config.hf_model_name)
    if args.dataset:
        return
    model = load_model(config)
    dataset = BaseDataset(path=config.dataset_path, experiment=config.experiment, model=model, start=config.dataset_start, end=config.dataset_end)

    experiments = []
    if args.logit_attribution:
        experiments.append(logit_attribution)
    if args.logit_lens:
        experiments.append(logit_lens)
    if args.ablate:
        experiments.append(ablate)
    if args.pattern:
        experiments.append(pattern)
    if args.all:
        experiments = [logit_attribution, logit_lens,  pattern, ablate]

    status = ["Pending" for _ in experiments]


    for i, experiment in enumerate(experiments):
        status[i] = "Running"
        table = display_experiments(experiments, status)
        console.print(table)
        experiment(model, dataset, config, args)
        status[i] = "Done"
    

if __name__ == "__main__":
    """
    Parses the command line arguments with ArgumentParser,
    then passes these to the main function.
    """
    config_defaults = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=config_defaults.model_name)
    parser.add_argument("--end", type=int, default=config_defaults.dataset_end)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--no-plot", dest="produce_plots", action="store_false", default=True
    )
    parser.add_argument("--batch", type=int, default=config_defaults.batch_size)
    parser.add_argument("--only-plot", action="store_true")
    parser.add_argument("--std-dev", action="store_true")

    parser.add_argument("--logit-attribution", action="store_true")
    parser.add_argument("--logit-lens", action="store_true")
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--total-effect", action="store_true")
    parser.add_argument("--pattern", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", action="store_true", default=False)
    parser.add_argument("--ablate-component", type=str, default="all")
    # Changed default to copyVSfact so it doesn't throw an error due to lack of folder arg
    parser.add_argument("--experiment", type=str, default="copyVSfact")
    parser.add_argument("--flag", type=str, default="")
    parser.add_argument("--normalize-logit", type=str, default='none')
    
    args = parser.parse_args()
    main(args)
