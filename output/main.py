import os
import json
dic = r'D:\training_complicating_data\output'
new_data = []
for file in os.listdir(dic):
    if file.endswith('.json'):
        with open(os.path.join(dic, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            if len(data) % 2 == 0:
                new_data.append(data)
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
