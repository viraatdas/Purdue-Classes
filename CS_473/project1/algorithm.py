import json


with open("allqueries.json") as json_file:
    data = json.load(json_file)
    
    num_relevant = 0
    number=  []

    for i in range(len(data['queries'])):
        curr = data['queries'][i]

        if "algorithm" in curr['text']:
            num_relevant+=1
            number.append(curr['number'])
