from datasets import load_dataset
import evaluate
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="/home/dyh/OKX/unsloth/qwen3-vl-2b-instruct-lora-sft-wikipedia-samples-outputs/checkpoint-1487",
    load_in_4bit=False
)

# FastVisionModel.for_inference(model)


wikipedia_datasets = load_dataset("BroAlanTaps/visual-wikipedia-en-samples")

train_dataset = wikipedia_datasets['train']
test_dataset = wikipedia_datasets['test']

# bleu_score = evaluate.load("bleu")
# rouge_score = evaluate.load("rouge")

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
        }
    ]
    return conversation

# converted_train_dataset = [convert_to_conversation(sample) for sample in train_dataset]
# converted_test_dataset = [convert_to_conversation(sample) for sample in test_dataset]

# for sample in test_dataset:
#     image_list = sample['images']
#     answer = sample['text']
#     conversation = convert_to_conversation(sample)
#     input_text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
#     inputs = tokenizer(
        
#     )

model.push_to_hub_merged("BroAlanTaps/wikipedia-visual-en-qwen2.5-2b-instruct-lora-sft-wikipedia-samples", tokenizer, token = "")