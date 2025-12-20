import json
with open("benign_train") as f:
    data = f.readlines()
train_count = 0
valid_count = 0
train_data_final = []
valid_data_final = []
for line in data:
    line = line.strip()
    train_data_final.append({
        "goal": line,
        "category": "benign",
        "target": "",
        "prompt": "",
        "id": train_count
    })
    train_count += 1

with open("benign_valid") as f:
    data = f.readlines()

for line in data:
    line = line.strip()
    valid_data_final.append({
        "goal": line,
        "category": "benign",
        "target": "",
        "prompt": "",
        "id": valid_count
    })

    valid_count += 1

with open("/fs/gamma-projects/audio/kajayaku/cmsc723/prompt_data.json") as f:
    data = json.load(f)

for item in data[:100]:
    item['category'] = 'malicious'
    item['id'] = train_count 
    train_count += 1
    train_data_final.append(item)

for item in data[100:110]:
    item['id'] = valid_count
    valid_count += 1
    item['category'] = 'malicious'
    valid_data_final.append(item)


with open("train_data_final.json", "w") as f:
    json.dump(train_data_final, f, indent=4)
with open("valid_data_final.json", "w") as f:
    json.dump(valid_data_final, f, indent=4)