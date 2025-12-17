from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments

model_name = "gogamza/kobart-base-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset("csv", data_files={"train": "./neg_5000.csv"})["train"]

# 빈 문장 제거
dataset = dataset.filter(
    lambda x: x["input"] is not None and x["output"] is not None and 
    len(str(x["input"]).strip()) > 0 and len(str(x["output"]).strip()) > 0
)

def preprocess(batch):
    inputs = [str(x) for x in batch["input"]]
    targets = [str(x) for x in batch["output"]]

    model_inputs = tokenizer(
        inputs,
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=64,
            truncation=True,
            padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./saved",
    per_device_train_batch_size=4,
    learning_rate=3e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("./version5_model")
tokenizer.save_pretrained("./version5_model")
