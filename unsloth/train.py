from unsloth import FastLanguageModel, FastModel
from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-2B-Instruct",
    load_in_4bit=False,
    use_gradient_checkpointing="unsloth",
)

# Load lora
model = FastVisionModel.get_peft_model(
    model=model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_mlp_modules=True,
    finetune_attention_modules=True,
    r=32,
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    random_state=2025,
)

from datasets import load_dataset

wikipedia_datasets = load_dataset("BroAlanTaps/visual-wikipedia-en-samples")

train_dataset = wikipedia_datasets['train']
test_dataset = wikipedia_datasets['test']

instruction = "Please identify and transcribe all text content in the images in sequence. If multiple images are provided, please transcribe them strictly in the order they appear, ensuring that the transcribed text of each subsequent image immediately follows the transcribed text of the previous image."


def convert_to_conversation(sample):
    user_input = []
    for img in sample["image"]:
        user_input.append({"type": "image", "image": img})
    
    user_input.append({"type": "text", "text": instruction})
    conversation = [
        {
            "role": "user",
            "content": user_input
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["text"]}]
        }
    ]
    return {"messages": conversation}

converted_train_dataset = [convert_to_conversation(sample) for sample in train_dataset]
converted_test_dataset = [convert_to_conversation(sample) for sample in test_dataset]

from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer, max_seq_length=10240000),
    train_dataset=converted_train_dataset,
    eval_dataset=converted_test_dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.01,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_steps=1,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=2025,
        weight_decay=0.001,
        output_dir="qwen3-vl-2b-instruct-lora-sft-wikipedia-samples-outputs",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=10,
        logging_dir="logs",
        report_to="swanlab",
        max_grad_norm=1.0,
        remove_unused_columns=False,
        max_seq_length=10240000

    ),
)

trainer.train()

# Save the model
model.save_pretrained("qwen3-vl-2b-instruct-lora-sft-wikipedia-samples-final")
tokenizer.save_pretrained("qwen3-vl-2b-instruct-lora-sft-wikipedia-samples-final")