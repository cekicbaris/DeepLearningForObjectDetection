import json
from config import MODELS

file_name = "stats/0821/stats.json"
summary_file_name = file_name[0:len(file_name)-5] + '_summary.json'




with open(file_name, 'r') as f:
    stat_file = json.load(f)


print("read")

summary = []

for stat in stat_file:
    duration = {}
    for model in MODELS:        
        duration['image_id'] = stat['name']
        duration[model['model_id']] = stat[model['model_id']]['stats']['duration']
    summary.append(duration)



with open(summary_file_name, 'w') as f:
    json.dump(summary , f)