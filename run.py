from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Use absolute path to avoid Hugging Face trying to fetch from Hub
model_path = Path(r"INSERT LOCATION OF YOUR MODEL THAT YOU HAVE DOWNLOADED AND EXTRACTED HERE.")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example prediction
example_code = """
#include <string.h>
#include <stdio.h>

int main() {
    char buf[10];
    strcpy(buf, "AAAAAAAAAAAAAAAAAAAAAAAA"); // potential overflow
    return 0;
}
"""

inputs = tokenizer(example_code, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
inputs = {k:v.to(device) for k,v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_label = torch.argmax(probs, dim=-1).item()
    label = model.config.id2label[pred_label]

print("🔎 Prediction:", label)
print("📊 Probabilities:", probs.cpu())
