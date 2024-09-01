import os
import json
dic = r"D:\multi_turn_training_data\output\we_need"
new_data = []
for file in os.listdir(dic):
    if file.endswith('.json'):
        with open(os.path.join(dic, file), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if len(data) % 2 == 0:
                    new_data.append(data)
            except:
                print(file)
with open('data_0828.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
