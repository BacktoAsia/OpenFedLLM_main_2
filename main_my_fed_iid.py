import torch
import copy
import os
from tqdm import tqdm
import numpy as np
import logging
import warnings

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
from vlmmodel.vlm import init_vlm_model,make_supervised_data_module,make_supervised_data_module_clients
warnings.filterwarnings("ignore", category=UserWarning)



def preprocess_logits_for_metrics(logits, labels):
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

script_args, fed_args, peft_config, model_args, data_args = get_config("./config/config_ai2_fed_iid.yaml")
training_args = get_training_args(script_args, script_args.learning_rate)
script_args = save_config(script_args, fed_args, model_args, data_args)


logdir = script_args.output_dir  
if not os.path.exists(logdir):
    os.makedirs(logdir)
logfile = os.path.join(logdir, 'training.log')
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("cuda: %s",torch.cuda.is_available())
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Current memory allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")

## ===== laod lamma =====
model,tokenizer=init_vlm_model(script_args,model_args, data_args)


# ===== Load the dataset =====
# ===== Split the dataset into clients =====
local_datasets, global_test_dataset,data_collator = make_supervised_data_module_clients(tokenizer=tokenizer, data_args=data_args,fed_args=fed_args) 

sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
logging.info("sample_num_list: %s",sample_num_list) 
training_args.max_steps=max(sample_num_list)//training_args.batch_size//4
script_args.max_steps=max(sample_num_list)//training_args.batch_size//4
logging.info("max_steps: %d",training_args.max_steps) 


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
local_dict_list = [copy.deepcopy(global_dict) for _ in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
if tokenizer.pad_token is None:
    # tokenizer.pad_token = tokenizer.unk_token   # following vicuna
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
## only for language model
# formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
# response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
# data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]

for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    logging.info(f">> ==================== Round {round+1} : {clients_this_round} ====================")
   
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)

        # ===== Train local model on the client side =====
        trainer = get_fed_local_vlm_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=local_datasets[client],
            eval_dataset=None,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
      
    #   显示当前模型占用的显存量
        logging.info(f"Current memory allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")

        results = trainer.train()
        training_loss[client].append(results.training_loss)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    set_peft_model_state_dict(model, global_dict)   # Update global model

    # ===== Test the model =====
   
    eval_results = trainer.evaluate(eval_dataset=global_test_dataset)
    logging.info(eval_results)
    with open(os.path.join(script_args.output_dir, "eval_results.txt"), "a") as f:
        f.write(f"Round {round+1}: {eval_results}\n")


    if (round+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))