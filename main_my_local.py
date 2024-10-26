import torch
import copy
import os
from tqdm import tqdm
import numpy as np

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
from vlmmodel.vlm import init_vlm_model,make_supervised_data_module_local
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_logits_for_metrics(logits, labels):
    # 只保留 logits 的类别信息
    predictions = torch.argmax(logits, dim=-1)
    return predictions, labels


def compute_metrics(eval_pred):
   
    labels = eval_pred.label_ids
    preds = eval_pred.predictions[0]

    all_labels = []
    all_preds = []

    for case_labels, case_preds in zip(labels, preds):
        case_lab,case_pred=[],[]
        for label, pred in zip(case_labels, case_preds):
            if label != -100:
                case_lab.append(label)
                case_pred.append(pred)
        all_labels.append(tokenizer.decode(case_lab))
        all_preds.append(tokenizer.decode(case_pred))


    right=0
    all_num=0
    for label_sentence, pred_sentence in zip(all_labels, all_preds):
        print(label_sentence)
        print(pred_sentence)
        label_is_pos=label_sentence.find("is")
        pred_is_pos=pred_sentence.find("is")
        if label_is_pos!=-1 and pred_is_pos!=-1:
            la=label_sentence[label_is_pos+3:label_is_pos+4]
            answer=pred_sentence[pred_is_pos+3:pred_is_pos+4]
            print("'"+la+"'","'"+answer+"'")
            if la==answer and la!=" ":
                right+=1
        all_num+=1
        print("================================")

    # print("================================")
   
    print(right,all_num)
    accuracy=right/all_num

    return {
        "accuracy": accuracy,
    }



##
# init_vlm_model() sets up the types of the vision tower, language model and tokenizer
# make_supervised_data_module() sets up the dataset and dataloader

# ===== Define the arguments =====
script_args, fed_args, peft_config, model_args, data_args = get_config("./config/config_ai2_local.yaml")
training_args = get_training_args(script_args, script_args.learning_rate)
script_args = save_config(script_args, fed_args, model_args, data_args)

## ===== laod lamma =====
model,tokenizer=init_vlm_model(script_args,model_args, data_args)

# ===== Load the dataset =====
local_datasets, global_test_dataset,data_collator = make_supervised_data_module_local(tokenizer=tokenizer, data_args=data_args,fed_args=fed_args) 


training_args.max_steps=len(local_datasets)//training_args.batch_size//4
script_args.max_steps=len(local_datasets)//training_args.batch_size//4
print(training_args.max_steps) 
print(os.path.join(script_args.output_dir, "eval_results.txt"))
# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()
    

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))

# ===== Define the tokenizer =====
if tokenizer.pad_token is None:
    # tokenizer.pad_token = tokenizer.unk_token   # following vicuna
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# print(tokenizer.pad_token_id)
# ===== Start federated training =====
training_loss = []

for round in tqdm(range(fed_args.num_rounds)):

    new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
    training_args = get_training_args(script_args, new_lr)

    # ===== Train local model on the client side =====
    trainer = get_local_vlm_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        local_dataset=local_datasets,
        eval_dataset=None,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        fed_args=fed_args,
        script_args=script_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,

    )
    
#   显示当前模型占用的显存量
    # print(f"Current memory allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")

    results = trainer.train()
    training_loss.append(results.training_loss)

  
    # ===== Save the model =====
   
    eval_results = trainer.evaluate(eval_dataset=global_test_dataset)
    print(eval_results)
    with open(os.path.join(script_args.output_dir, "eval_results.txt"), "a") as f:
        f.write(f"Round {round+1}: {eval_results}\n")


    if (round+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))