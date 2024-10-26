import torch
from transformers import Trainer
from utils import *
from federated_learning import *
from test_config import get_config
from vlmmodel.vlm import init_vlm_model_test,make_supervised_data_module_test
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    precision, recall, f1, acc = calculate_matching_accuracy(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

test_config="/home/weiying/OpenFedLLM-main/output1/scienceQA_20000_fedavg_c10s5_i10_b4a1_l1024_r8a32_20241010012832/args.json"
model_path="/home/weiying/OpenFedLLM-main/output1/scienceQA_20000_fedavg_c10s5_i10_b4a1_l1024_r8a32_20241010012832/checkpoint-10"


# ===== Define the arguments =====
script_args, model_args, data_args = get_config(test_config)
training_args = script_args

# Load the model for evaluation
model, tokenizer = init_vlm_model_test(training_args, model_args,model_path)
# model.load_state_dict(torch.load(model_path))  # Load the model state dictionary from the given path
model.to(device)
model.eval()  # Set model to evaluation mode

# Load the global test dataset for evaluation
_, global_test_dataset, data_collator = make_supervised_data_module_test(tokenizer=tokenizer, data_args=data_args)

# Define the trainer for evaluation
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Evaluate the model on the global test dataset
eval_results = trainer.evaluate(eval_dataset=global_test_dataset)

# Process and print the evaluation results
print("Evaluation results:")
for key, value in eval_results.items():
    print(f"{key}: {value}")

# Save the evaluation results if needed
# np.save(os.path.join(script_args.output_dir, "evaluation_results.npy"), eval_results)