from unsloth import FastModel
import torch

# [NEW] Note: To use the COMET and BLEURT metrics, you need to install them first:
# pip install unbabel-comet
# pip install evaluate bleurt
# pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
# The first time you run this, it will download the models (e.g., wmt22-comet-da, bleurt-20).

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    max_seq_length = 256, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 64,           # Larger = higher accuracy, but might overfit
    lora_alpha = 64,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

from datasets import load_dataset
# Load the full dataset
train_dataset = load_dataset("Thermostatic/NeuralTranslate-mt-en-es-v2-Splits", split="train")
validation_dataset = load_dataset("Thermostatic/NeuralTranslate-mt-en-es-v2-Splits", split="validation")

print(f"Dataset splits:")
print(f"  Train: {len(train_dataset)} examples")
print(f"  Validation: {len(validation_dataset)} examples")

# 2. Standardize and format all three dataset splits
from unsloth.chat_templates import standardize_data_formats

train_dataset = standardize_data_formats(train_dataset)
validation_dataset = standardize_data_formats(validation_dataset)

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

# Apply formatting to all splits
train_dataset = train_dataset.map(
    formatting_prompts_func,
    batched = True
    )
validation_dataset = validation_dataset.map(
    formatting_prompts_func,
    batched = True
    )

from evaluate import load
import numpy as np # Make sure to import numpy
# [NEW] Import custom BLEURT classes
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer


# --- Load all evaluation metrics once ---
chrf_metric = load("chrf")
comet_metric = load("comet")

# [NEW] Load the custom BLEURT-20 model and tokenizer
print("Loading custom BLEURT-20 model...")
bleurt_model_name = 'lucadiliello/BLEURT-20'
bleurt_config = BleurtConfig.from_pretrained(bleurt_model_name)
bleurt_model = BleurtForSequenceClassification.from_pretrained(bleurt_model_name)
bleurt_tokenizer = BleurtTokenizer.from_pretrained(bleurt_model_name)

# Move BLEURT model to GPU if available for faster evaluation
bleurt_device = "cuda" if torch.cuda.is_available() else "cpu"
bleurt_model.to(bleurt_device)
bleurt_model.eval() # Set to evaluation mode
print("BLEURT-20 model loaded successfully.")
# ----------------------------------------


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def compute_metrics(eval_pred):
    pred_ids, label_ids = eval_pred

    # Clean up predictions and labels
    pred_ids = np.where(label_ids != -100, pred_ids, tokenizer.pad_token_id)
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    # Decode predictions and labels.
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # --- COMET requires the original source text ---
    # We extract it from the validation_dataset, which is accessible in this scope.
    # After `standardize_data_formats`, the key changes from 'value' to 'content'.
    sources = [
        ex["conversations"][0]["content"]
        for ex in validation_dataset
    ]

    # --- Calculate chrF score ---
    decoded_labels_nested = [[label] for label in decoded_labels]
    chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels_nested)

    # --- Calculate COMET score ---
    comet_result = comet_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        sources=sources
    )

    # --- [NEW] Calculate BLEURT score using the custom model ---
    with torch.no_grad():
        # The custom tokenizer expects (references, candidates)
        inputs = bleurt_tokenizer(decoded_labels, decoded_preds, padding='longest', return_tensors='pt').to(bleurt_device)
        bleurt_logits = bleurt_model(**inputs).logits.flatten().tolist()
    bleurt_score = np.mean(bleurt_logits)


    # Return a dictionary of all scores.
    return {
        "chrf": chrf_result["score"],
        "comet": comet_result["mean_score"],
        "bleurt": bleurt_score,
    }


# 3. Configure the trainer to use the validation set and report metrics
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = validation_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_ratio = 0.1,
        num_train_epochs = 10, # Set this for 1 full training run.
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "wandb", # Use this for WandB etc
        dataset_num_proc=4,  # Use more processes for mapping
        prediction_loss_only=False,
        # New arguments for validation and saving
        eval_strategy="epoch",
        save_strategy = "epoch",
        load_best_model_at_end=True,
        metric_for_best_model="comet", # [NEW] You can now change this to "bleurt"
        greater_is_better=True,
    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

# Start training. Validation loss and metrics (chrF, COMET, BLEURT) will be calculated and reported.
print("Starting training...")
trainer_stats = trainer.train()

# --- The rest of the script for inference and saving remains the same ---

print("\n--- Example Inference ---")
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Traduce al español: Auh in ye yuhqui in on tlenamacac niman ye ic teixpan on motlalia ce tlacatl itech mocaua.",}]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)