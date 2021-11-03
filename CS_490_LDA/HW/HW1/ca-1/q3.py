# file q3.py
print("===[Q3:START]===") 

import re
from collections import defaultdict
import heapq

filename = 'LDA_graph_nodes.csv'

interests_re = re.compile(r'\|?([^\|\n-]+)\|?\n?')

num_entries = 6
max_heap_cap = 10

interests_freq = defaultdict(lambda: 0)

def clean_html_tokens(raw_string):
    reg = re.compile(r'\%\d\d')
    return re.sub(reg, '', raw_string)


with open(filename, 'r', encoding='utf-8-sig') as f:
    # process file here
    for line in f:
        fields = line.split(",")
        if len(fields) != num_entries:
            continue

        node_id = fields[0]
        
        matches = interests_re.findall(fields[5])

        clean_matches = [string.lower() for string in matches]
        clean_matches = [clean_html_tokens(raw_string) for raw_string in clean_matches]

        for string in clean_matches:
            interests_freq[string] += 1
        

    h = [] 
    for interest in interests_freq:
        heapq.heappush(h, (-interests_freq[interest], interest))
    
    # pop the 10 elements in the heap so that the most popular element is in the beginning
    interests = []
    for _ in range(max_heap_cap):
        interests.append(heapq.heappop(h)[1])

    # go back to the beginning of the file
    f.seek(0)

    for line in f:
        fields = line.split(",")
        if len(fields) != num_entries:
            continue

        node_id = fields[0]
        
        matches = interests_re.findall(fields[5])

        clean_matches = [string.lower() for string in matches]
        clean_matches = [clean_html_tokens(raw_string) for raw_string in clean_matches]

        # find common words between clean matches and interests
        common_words = set(clean_matches) & set(interests)
        
        final_list = []
        for interest in interests:
            if interest in common_words:
                final_list.append(interest)
                common_words.remove(interest)

        print(f'{node_id}\t{final_list}')


print("===[Q3:END]===")

