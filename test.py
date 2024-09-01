import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
with open(r'D:\multi_turn_training_data\data\cross_validation_10.jsonl', 'r', encoding='utf-8') as f:
    data_inst_structure = [json.loads(line)["instruction"] for line in f]
print(data_inst_structure)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer_relation = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
model_relation = AutoModelForSequenceClassification.from_pretrained(
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7").to(device)
def get_label(premise, hypothesis):
    input = tokenizer_relation(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
    output = model_relation(input["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = [1, 2, 3]
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
    max_label = max(prediction, key=prediction.get)
    return max_label
n = len(data_inst_structure)
result = [[0] * n for _ in range(n)]
for i in tqdm(range(n)):
    for j in range(i + 1, len(data_inst_structure)):
        result[i][j] = get_label(data_inst_structure[i], data_inst_structure[j])
with open(r'D:\multi_turn_training_data\data\relation_10.txt', 'w', encoding='utf-8') as f:
    for row in result:
        f.write('\t'.join([str(x) for x in row]) + '\n')