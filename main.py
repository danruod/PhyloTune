import argparse
import torch
from eval_taxa import main as eval_taxa
from phylotune import main as phylotune
from attn_analysis import main as attn_analysis
from utils.dataset import SequenceDatasetMultiRank              

    
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

                        
TASK_CONFIGS = {
    "PhyloTune": {
        "description": "Inference for new sequences",
        "arguments": [
            {"name": "--seqs", "kwargs": {"required": True, "type": str, "nargs": '+', "help": "New sequences"}},
            {"name": "--marker", "kwargs": {"required": True, "type": str, "help": "Marker"}},
            {"name": "--threshold", "kwargs": {"required": False, "type": float, "help": "Threshold for novelty score"}},
            {"name": "--num_segs", "kwargs": {"required": False, "type": int, "default": 3, "help": "Number of equal segments to divide the sequence into."}},
            {"name": "--k", "kwargs": {"required": False, "type": int, "default": 1, "help": "K segments with high attention scores constitute high-attn regions."}},
        ],
        "func": phylotune,
    },
    "eval_taxa": {
        "description": "Evaluate Taxonomic Units",
        "arguments": [
            {"name": "--batch_size", "kwargs": {"required": False, "type": int, "default": 64, "help": "Batch size"}},
            {"name": "--exp_name", "kwargs": {"required": False, "type": str, "default": "test", "help": "used to file name"}},
            {"name": "--return_markers", "kwargs": {"required": False, "action": "store_true", "help": "return results for each markers, Only valid when dataset is Plant"}},
            {"name": "--plot", "kwargs": {"required": False, "action": "store_true", "help": "plot confusion matrix,Only valid when dataset is Plant"}},
            
        ],
        "func": eval_taxa,
    },
    "attn_analysis": {
        "description": "Analysis",
        "arguments": [
            {"name": "--gap_thres", "kwargs": {"required": False, "type": float, "default": 0.9, "help": "filter loci with too much gap"}},
            {"name": "--metrics", "kwargs": {"required": True, "type": str, "nargs": '+', "help": "Metrics to evaluate"}},
            {"name": "--ignore_gap", "kwargs": {"required": False, "action": "store_true", "help": "Ignore gap"}},
            {"name": "--plot", "kwargs": {"required": False, "action": "store_true", "help": "Plot or not"}},
        ],
        "func": attn_analysis,
    },
}

COMMON_ARGS = [
    {"name": "--dataset", "kwargs": {"required": True, "type": str, "choices": ["Plant", "Bordetella"], "help": "only support Plant and bordetella"}},
    {"name": "--data_path", "kwargs": {"required": False, "type": str, "default": "./datasets", "help": "Path for data"}},
    {"name": "--model_path", "kwargs": {"required": False, "type": str, "default": "./checkpoints", "help": "Path for model"}},
    {"name": "--result_path", "kwargs": {"required": False, "type": str, "default": "./results", "help": "Path for saving results"}},
]
                        
def main():
    parser = argparse.ArgumentParser(description="Task Manager")
    subparsers = parser.add_subparsers(dest="task_type", required=True, help="Type of task to execute")
    
    for task_name, config in TASK_CONFIGS.items():
        task_parser = subparsers.add_parser(task_name, description=config["description"], help=config["description"])
        
        for arg in config["arguments"]:
            task_parser.add_argument(arg["name"], **arg["kwargs"])
        
        for common_arg in COMMON_ARGS:
            task_parser.add_argument(common_arg["name"], **common_arg["kwargs"])
        
        task_parser.set_defaults(func=config["func"])
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
    