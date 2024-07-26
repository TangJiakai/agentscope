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

from llmtuning.code.utils.constants import *
from llmtuning.code.utils.utils import check_load_adapter

wandb.init(mode="disabled")
tqdm.pandas()


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
        output_dir=SAVE_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        max_seq_length=2048,
    )

    if_load_adapter = check_load_adapter()
    if if_load_adapter:
        model = AutoModelForCausalLM.from_pretrained(
            LLM_DIR_PATH,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16, 
        )
        model = PeftModel.from_pretrained(model, SAVE_DIR)
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

    trainer.train()
    print("SFT-Training completed!")
    return model


def ppo_train(model, tokenizer):
    dataset = load_dataset(PPO_FILE_PATH, data_files="ppo_data.json", split="train")
    def tokenize(sample):
        sample["query_ids"] = tokenizer.encode(sample["prompt"])
        sample["response_ids"] = tokenizer.encode(sample["completion"])
        sample["query"] = tokenizer.decode(sample["query_ids"])
        sample["response"] = tokenizer.decode(sample["response_ids"])
        return sample
    dataset = dataset.map(tokenize, batch_size=False)
    dataset.set_format(type="torch", columns=["query_ids", "response_ids", "reward"])

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model,
        device_map="auto",
        torch_dtype=torch.bfloat16, 
    )
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    ref_model = create_reference_model(model)
    
    ppo_config = PPOConfig(
        whiten_rewards=True,
        batch_size=len(dataset),
        remove_unused_columns=False,
        mini_batch_size=2,
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
        query_tensors = batch["query_ids"]
        response_tensors = batch["response_ids"]
        rewards_tensors = [x.float() for x in batch["reward"]]

        stats = trainer.step(query_tensors, response_tensors, rewards_tensors)

    trainer.save_pretrained(SAVE_DIR)
    print(f"RLHF-Training completed! Saving the model to {SAVE_DIR}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_DIR_PATH,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = sft_train(tokenizer)
    ppo_train(model, tokenizer)


if __name__ == "__main__":
    main()