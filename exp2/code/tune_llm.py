import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
from datasets import load_dataset
from trl import (
    SFTConfig, 
    SFTTrainer, 
    DataCollatorForCompletionOnlyLM, 
    PPOConfig, 
    PPOTrainer, 
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from peft import LoraConfig, PeftModel, get_peft_model
import torch
import wandb
import argparse

from utils.constants import *
from utils.utils import check_load_adapter, check_dirs

wandb.init(mode="disabled")
tqdm.pandas()
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def parse_args() -> argparse.Namespace:
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning_mode", type=str, help="sft or ppo", default="sft")
    return parser.parse_args()


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"### Question: {example['prompt'][i]}\n\n### Assistant: {example['completion'][i]}"
        output_texts.append(text)
    return output_texts


def sft_train(tokenizer):
    dataset = load_dataset(SFT_FILE_PATH, data_files='sft_data.json', split="train")

    response_template_string = "\n### Assistant:"
    response_template_ids = tokenizer.encode(response_template_string, add_special_tokens=False)[2:]

    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    sft_config = SFTConfig(
        output_dir=TMP_SAVE_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        max_seq_length=8192,
    )

    if_load_adapter = check_load_adapter()
    if if_load_adapter:
        model = AutoModelForCausalLM.from_pretrained(
            LLM_DIR_PATH,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16, 
        )
        model = PeftModel.from_pretrained(model, TMP_SAVE_DIR)
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
    else:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = AutoModelForCausalLM.from_pretrained(
            LLM_DIR_PATH,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16, 
        )
        model = get_peft_model(model, peft_config)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        tokenizer=tokenizer,
        data_collator=collator,
        formatting_func=formatting_prompts_func,
    )
    
    print("Starting SFT training")
    trainer.train()

    trainer.save_model(TMP_SAVE_DIR)
    return model


def ppo_train(tokenizer):
    dataset = load_dataset(PPO_FILE_PATH, data_files="ppo_data.json", split="train")
    def tokenize(sample):
        sample["query_ids"] = tokenizer.encode(sample["prompt"])
        sample["response_ids"] = tokenizer.encode(sample["completion"])
        sample["query"] = tokenizer.decode(sample["query_ids"])
        sample["response"] = tokenizer.decode(sample["response_ids"])
        sample['reward'] = float(sample['reward'])
        return sample
    dataset = dataset.map(tokenize, batch_size=False)
    dataset.set_format(type="torch", columns=["query_ids", "response_ids", "reward"])

    if_load_adapter = check_load_adapter()
    if if_load_adapter:
        model = AutoModelForCausalLM.from_pretrained(
            LLM_DIR_PATH,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16, 
        )
        model = PeftModel.from_pretrained(model, TMP_SAVE_DIR)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model,
            device_map="auto",
            torch_dtype=torch.bfloat16, 
        )
    else:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            LLM_DIR_PATH,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16, 
            peft_config=peft_config,
        )

    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    ref_model = create_reference_model(model)
    
    ppo_config = PPOConfig(
        ppo_epochs=1,
        whiten_rewards=True,
        batch_size=len(dataset),
        remove_unused_columns=False,
        mini_batch_size=1,
    )

    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    for _epoch, batch in tqdm(enumerate(trainer.dataloader)):
        print(f"The batch size is {len(batch['reward'])}...")
        query_tensors = batch["query_ids"]
        response_tensors = batch["response_ids"]
        rewards_tensors = [x.float() for x in batch["reward"]]

        stats = trainer.step(query_tensors, response_tensors, rewards_tensors)

    trainer.save_pretrained(TMP_SAVE_DIR)


def copy_saves():
    os.system(f"cp -r {TMP_SAVE_DIR} {SAVE_DIR}")
    os.system(f"rm -r {TMP_SAVE_DIR}")
    print(f"Copied the trained model to {SAVE_DIR}")


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_DIR_PATH,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if args.tuning_mode == "sft":
        if os.path.isdir(SFT_FILE_PATH) and os.listdir(SFT_FILE_PATH):
            sft_train(tokenizer)
            print(f"SFT-Training completed! Saving the model to {SAVE_DIR}")
    elif args.tuning_mode == "ppo":
        if os.path.isdir(PPO_FILE_PATH) and os.listdir(PPO_FILE_PATH):
            ppo_train(tokenizer)
            print(f"RLHF-Training completed! Saving the model to {SAVE_DIR}")
    else:
        raise ValueError(f"Invalid tuning mode {args.tuning_mode}")

if __name__ == "__main__":
    args = parse_args()
    check_dirs()
    main(args)
    copy_saves()