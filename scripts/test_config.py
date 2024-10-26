from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import HfArgumentParser
import transformers
import yaml


# Define and parse arguments.

@dataclass
class ModelArguments:
    llm_type: Optional[str] = field(default="")

    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    pad_token_version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_tower_type: Optional[str] = field(default='clip')


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    
    dataset_name: Optional[str] = field(
        default="lucasmccabe-lmi/CodeAlpaca-20k", metadata={"help": "the dataset name"}
    )
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "the local data directory if you want to use downloaded data"})



@dataclass
class ScriptArguments():
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
   
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})

    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})

    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})


    
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    seed: Optional[int] = field(default=2023, metadata={"help": "the seed to use"})

    

      

import json
def get_config(test_config_path):
    # 读取 YAML 配置文件
    with open(test_config_path, 'r') as stream:
        try:
            config = json.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
  
    valid_keys = [field.name for field in DataArguments.__dataclass_fields__.values()]
    filtered_config = {key: value for key, value in config["data_args"].items() if key in valid_keys}
    data_args = DataArguments(**filtered_config)

    valid_keys = [field.name for field in ModelArguments.__dataclass_fields__.values()]
    filtered_config = {key: value for key, value in config["model_args"].items() if key in valid_keys}
    model_args = ModelArguments(**filtered_config)

    valid_keys = [field.name for field in ScriptArguments.__dataclass_fields__.values()]
    filtered_config = {key: value for key, value in config["script_args"].items() if key in valid_keys}
    script_args = ScriptArguments(**filtered_config)


    print(model_args)
    return script_args, model_args, data_args

# ===== Define the training arguments =====



