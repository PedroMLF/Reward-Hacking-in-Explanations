#ADAPTED FROM: https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/merge_peft_adapter.py

import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser


import env_setup

env_setup.setup_env(use_temp=False)


@dataclass
class ScriptArguments:
    """
    The input names representing the Adapter and Base model fine-tuned with PEFT, and the output name representing the
    merged model.
    """

    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the adapter name"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the base model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the merged model name"})


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    assert script_args.adapter_model_name is not None, "please provide the name of the Adapter you would like to merge"
    assert script_args.base_model_name is not None, "please provide the name of the Base model"
    assert script_args.output_name is not None, "please provide the output name of the merged model"

    # Resolve adapter model name
    adapter_model_name = glob(script_args.adapter_model_name)
    assert len(adapter_model_name) == 1
    adapter_model_name = adapter_model_name[0]

    peft_config = PeftConfig.from_pretrained(adapter_model_name)
    if peft_config.task_type == "SEQ_CLS":
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.base_model_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            cache_dir=os.environ.get("HF_MODELS_CACHE_DIR"),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.base_model_name,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            cache_dir=os.environ.get("HF_MODELS_CACHE_DIR"),
        )

    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)

    # Load the PEFT model
    model = PeftModel.from_pretrained(model, adapter_model_name)
    model.eval()

    model = model.merge_and_unload()

    model.save_pretrained(f"{script_args.output_name}")
    tokenizer.save_pretrained(f"{script_args.output_name}")
