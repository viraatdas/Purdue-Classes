# file q1.py
import time

class BloomFilter:
    def __init__(self):
        self.bitstring = int(0)

    def hash(self, str_input):
        # this conversion from string to integers, surprisingly works even if string has less than 4 characters
        int_input = int.from_bytes(bytes(str_input[0:4]),"big") 

        return ( 0xFFFFFFFFFFFFFFFF & (int_input << 5) )

    def add(self, key_input):
        hash_val = self.hash(key_input)
        self.bitstring = self.bitstring ^ hash_val 

    def __contains__(self, key_input):
        hash_val = self.hash(key_input)
        return (self.bitstring & hash_val == hash_val)

print("===[Q1:START]===") 

user_data = {}

def read_file():
    filename = 'LDA_graph_nodes.csv'
    # filename = 'nodes_test.csv'

    with open(filename, 'r', encoding='utf-8-sig') as f:
        # process file here
        for line in f:
            fields = line.strip().split(",")
            if len(fields) == 6:
                node_id = int(fields[0])
                fields[5] = fields[5].split("|")
                # create the Bloom filter for user interests
                bf = BloomFilter()
                # for each interest, add interest to Bloom filter
                for interest in fields[5]:
                    # +"    " makes sure string has at least 4 bytes
                    bf.add((interest+"    ").encode("utf-8"))
                # define user_data as a tuple (BloomFilter, other-fields)
                user_data[node_id] = (bf,fields[1:])

def query_user_data(interest_key):
    # will count all positive instances that our Bloom filter returns
    cnt_all_positives = 0
    # will count all true positive instances
    cnt_true_positives = 0
    # must define an encoding of the string in order to transform it into a number
    # "    " makes sure string has at least 4 bytes
    utf8_interest_key = (interest_key+"    ").encode("utf-8")

    for id in user_data:
        bf, fields = user_data[id]
        # check if interest is in the user's Bloom filter
        if utf8_interest_key in bf:
            cnt_all_positives += 1
            # if Bloom filter says it could be positive...
            for interests in fields[4]:
                if interests == interest_key:
                    cnt_true_positives += 1
                    print(f'User {id} is interested in {fields[4]} including {interest_key}')

    print(f'\nOut of {len(user_data)} instances, '\
          f'we found false positives = {cnt_all_positives-cnt_true_positives} and '\
          f'true positives = {cnt_true_positives}')
# MAIN
# Read file and get hash table user_data ready
read_file()
for interest in ["AI","computers","travel"]:
    print(f'\n##### Interest {interest} ######')
    # Query all users for a given interest
    start_time = time.time()
    query_user_data(interest_key=interest)
    end_time = time.time()
    print(f'Execution time: {(end_time - start_time)*1000} ms\n')

print("===[Q1:END]===")

