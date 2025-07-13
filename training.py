from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from score_generation import score
from Generate_sentence import add_sentence



def model_create():
    
    model_id =  "distilbert-base-uncased"  # Example model, replace with your choice
    # For regression, you might want to use a model like "distilbert-base-
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=1,  # Should be 1 for regression
    problem_type="regression"
    )

    return tokenizer, model



def preprocess(example):
    model,tokenizer=model_create()
    encoding = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    encoding["labels"] = float(example["score"])  # Regression label should be a float
    return encoding

def train():
    tokenizer, model = model_create()
    args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8, # Increased batch size for efficiency
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",
    # evaluation_strategy="epoch" # Removed evaluation strategy
    )
    df = add_sentence()  # Generate sentences
    df['text'] = df['text'].apply(lambda x: x.strip('"'))
    df['facet'] = df['facet'].apply(lambda x: x.strip('"'))
    df_scored = score(df)
    dataset = Dataset.from_pandas(df_scored)
    tokenized_dataset = dataset.map(preprocess)

    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset, # Using the same dataset for eval for simplicity
    )

    trainer.train()
    trainer.save_model("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    return model, tokenizer