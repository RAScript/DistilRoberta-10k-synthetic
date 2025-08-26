# DistilRoBERTa Stack/Heap Overflow Detector Training Script 

# --- Step 1: Install dependencies ---
!pip install transformers datasets torch scikit-learn -q

# --- Step 2: Generate synthetic dataset ---
import pandas as pd

vuln_classes = ["stack_buffer_overflow", "heap_overflow"]

def generate_snippet(vuln_type, i):
    if vuln_type == "stack_buffer_overflow":
        return f"void f() {{ char buf[10]; gets(buf); // {vuln_type} #{i} }}"
    elif vuln_type == "heap_overflow":
        return f"void f() {{ char *buf = malloc(10); strcpy(buf, 'AAAAAAAAAAAAAAAA'); // {vuln_type} #{i} }}"

data = []
for vuln in vuln_classes:
    for i in range(5000):  # 5k per type
        snippet = generate_snippet(vuln, i)
        data.append({"code": snippet, "label": vuln})

df = pd.DataFrame(data)

# Split into train/test (80/20)
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Dataset generated: {len(train_df)} training, {len(test_df)} testing samples")

# --- Step 3: Load dataset ---
from datasets import load_dataset
dataset = load_dataset("csv", data_files={"train":"train.csv", "test":"test.csv"})

# --- Step 4: Encode labels ---
labels = vuln_classes
label2id = {label:i for i,label in enumerate(labels)}
id2label = {i:label for i,label in enumerate(labels)}

def encode_labels(example):
    example["label"] = label2id[example["label"]]
    return example

dataset = dataset.map(encode_labels)

# --- Step 5: Tokenize ---
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

def tokenize(batch):
    return tokenizer(batch["code"], padding="max_length", truncation=True, max_length=256)

dataset = dataset.map(tokenize, batched=True)

# --- Step 6: Load model ---
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# --- Step 7: Training setup ---
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    report_to="none",
    num_train_epochs=6,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch"
)

# --- Step 8: Metrics ---
from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(eval_pred):
    logits, labels_eval = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels_eval, preds),
            "f1": f1_score(labels_eval, preds, average="weighted")}

# --- Step 9: Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# --- Step 10: Train with Timer ---
import time
start = time.time()
trainer.train()
end = time.time()
print(f"⏱ Training took {(end-start)/60:.2f} minutes")

# --- Step 11: Save and Zip Model ---
trainer.save_model("./distilroberta-vuln-detector")
tokenizer.save_pretrained("./distilroberta-vuln-detector")
!zip -r model.zip ./distilroberta-vuln-detector

# --- Step 12: Download ---
from google.colab import files
files.download("model.zip")
