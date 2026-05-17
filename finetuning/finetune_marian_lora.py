# =========================
# 1. Install dependencies
# =========================
!pip install -q -U transformers datasets accelerate sentencepiece sacrebleu evaluate peft

# =========================
# 2. Imports
# =========================
import torch
from datasets import load_dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =========================
# 3. Paths
# =========================
csv_path = "/content/drive/MyDrive/MT/human_dataset_trainable.csv" #Change to human/llm/nmt
output_dir = "/content/drive/MyDrive/opus_human_lora" #Change to human/llm/nmt

# =========================
# 4. Load dataset
# =========================
dataset = load_dataset("csv", data_files={"data": csv_path})["data"]

# Keep only non-empty rows
def keep_valid(example):
    src = str(example["src"]).strip() if example["src"] is not None else ""
    tgt = str(example["tgt"]).strip() if example["tgt"] is not None else ""
    return len(src) > 0 and len(tgt) > 0

dataset = dataset.filter(keep_valid)

# Simple split for now
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
valid_ds = dataset["test"]

print("Train size:", len(train_ds))
print("Valid size:", len(valid_ds))
print(train_ds[0])

# =========================
# 5. Load base model/tokenizer
# =========================
model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name, use_safetensors=False)

# Optional but sometimes helpful for Marian warnings
model.config.tie_word_embeddings = False

# =========================
# 6. Tokenization
# =========================
max_source_length = 256
max_target_length = 256

def preprocess_function(examples):
    inputs = [str(x).strip() for x in examples["src"]]
    targets = [str(x).strip() for x in examples["tgt"]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
    )

    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
    )

    # Ignore padding in loss
    label_ids = labels["input_ids"]
    label_ids = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
        for seq in label_ids
    ]

    model_inputs["labels"] = label_ids
    return model_inputs

tokenized_train = train_ds.map(
    preprocess_function,
    batched=True,
    remove_columns=train_ds.column_names
)

tokenized_valid = valid_ds.map(
    preprocess_function,
    batched=True,
    remove_columns=valid_ds.column_names
)

print(tokenized_train[0])

# =========================
# 7. Add LoRA
# =========================
# Marian attention layers usually expose q_proj and v_proj
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# 8. Data collator
# =========================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# =========================
# 9. Training args
# =========================
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,

    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=10,

    learning_rate=3e-4,
    warmup_ratio=0.05,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,

    num_train_epochs=6,

    weight_decay=0.0,
    max_grad_norm=1.0,

    predict_with_generate=False,

    bf16=use_bf16,
    fp16=not use_bf16,

    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)

# =========================
# 10. Trainer
# =========================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    processing_class=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# =========================
# 11. Train
# =========================
train_result = trainer.train()
print(train_result)

# =========================
# 12. Save LoRA adapter
# =========================
adapter_path = output_dir + "/final_adapter"
trainer.save_model(adapter_path)
tokenizer.save_pretrained(adapter_path)

print("Saved adapter to:", adapter_path)

# =========================
# 13. Quick generation test
# =========================
test_text = "Alice was beginning to get very tired of sitting by her sister on the bank."

inputs = tokenizer(
    test_text,
    return_tensors="pt",
    truncation=True,
    max_length=max_source_length
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4
    )

print("Translation:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
