import pandas as pd
import json
import numpy as np
from collections import defaultdict, OrderedDict
from operator import add


dishes_df = pd.read_csv("/homes/cs473/project2/data/dishes.csv")
with open("/homes/cs473/project2/data/user_ratings_train.json") as rating_train_file:
    ratings_train_dict = json.load(rating_train_file)


num_dishes = len(dishes_df)
user_dishes_rating_list = [[]] * len(ratings_train_dict)

for user in ratings_train_dict:
    dish_list = [-1] * num_dishes
    sum_rating = 0
    num_rating = 0
    for dish_id, actual_rating in ratings_train_dict[user]:
        actual_rating = float(actual_rating)
        dish_id = int(dish_id)
        dish_list[dish_id] = actual_rating
        sum_rating += actual_rating
        num_rating += 1
    avg = sum_rating/num_rating
    for i in range(len(dish_list)):
        if dish_list[i] != -1:
            dish_list[i] = dish_list[i] - avg
    dish_list.append(avg)
    user_dishes_rating_list[int(user)] = dish_list

col_names = [f'{i}' for i in range(num_dishes)] + ["user_avg"]
sub_mean_user_dishes_rating_df = pd.DataFrame(user_dishes_rating_list, columns=col_names)



# def cos_sim(a, b):
#     a = np.asarray(a)
#     b = np.asarray(b)
    
#     dot_product = np.dot(a,b)
#     norm_a = np.linalg.norm(a)
#     norm_b = np.linalg.norm(b)
#     return dot_product/(norm_a * norm_b)

# similiarity_between_users = defaultdict(lambda: [0] * len(ratings_train_dict))

# def fill_similarity_table():
#     for index1, a in sub_mean_user_dishes_rating_df.iterrows():
#         a = a.tolist()[:-1]
#         for index2, b in sub_mean_user_dishes_rating_df.iterrows():
#             if index1 == index2:
#                 continue

#             b = b.tolist()[:-1]
#             sim_score = cos_sim(a, b)
#             similiarity_between_users[index1][index2] = sim_score


with open("/homes/cs473/project2/data/user_ratings_test.json") as rating_test_file:
    ratings_test_dict = json.load(rating_test_file)


import dill

'''
Takes very long to fill the similarity table so we'll serialize it 
and then load it up 
'''
# fill_similarity_table()
# with open('similarity_between_users.pkl', 'wb+') as file:
#     dill.dump(similiarity_between_users, file)



with open('similarity_between_users.pkl', 'rb') as handle:
    similiarity_between_users = dill.load(handle)


def predict(user, dish_id):
    user = int(user)
    if sub_mean_user_dishes_rating_df.loc[user][dish_id] != 0:
        return sub_mean_user_dishes_rating_df.loc[user][dish_id]
  
    user_avg = sub_mean_user_dishes_rating_df.loc[int(user)]['user_avg']
    summation = 0
    sum_similarity_score = 0
    for index, row in sub_mean_user_dishes_rating_df.iterrows():
        if index == user:
            continue
        sim_value = similiarity_between_users[user][index]
        summation += row[dish_id] * sim_value
        sum_similarity_score += sim_value

    return user_avg + summation/sum_similarity_score

def get_top_n_docs(n, user, like_set, not_like_set):
    user = int(user)
    # list of cosine similarity scores
    vals = similiarity_between_users[user]

    dishes_user_not_rated = set()

    for dish_id, val in enumerate(sub_mean_user_dishes_rating_df.values[user]):
        if val == -1:
            dishes_user_not_rated.add(dish_id)
    # these are values that we should predict
    # if a dish doesn't occur here that means ignore the values
    seen_test_set = like_set | not_like_set

    cosine_score_index = {score: i for i, score in enumerate(vals)}
    top_dishes = []
    
    for score, index in sorted(cosine_score_index.items(), reverse=False):
        curr_user_values = sub_mean_user_dishes_rating_df.values[index]
        for dish_id in np.argsort(curr_user_values)[::-1]:
            if curr_user_values[dish_id] == -1 or dish_id not in dishes_user_not_rated:
                continue
            if dish_id in seen_test_set:
                top_dishes.append(dish_id)
        
            if len(top_dishes) >= n:
                break
    return top_dishes[:n]

n = 10
total10 = defaultdict(list)
total20 = defaultdict(list)
def calculate_recall(n, relevant_documents, retrieved_documents):
    
    retrieved_documents = set(retrieved_documents[0])
    return len(relevant_documents.intersection(retrieved_documents))/len(relevant_documents)

def calculate_precision(n, relevant_documents, retrieved_documents):

    retrieved_documents = set(retrieved_documents[0])
    return len(relevant_documents.intersection(retrieved_documents))/len(retrieved_documents)

avg_precision10 = []
avg_recall10 = []

avg_precision20 = []
avg_recall20 = []
for user in ratings_test_dict:
    test_user_seen_like = defaultdict(set)
    test_user_seen_not_like = defaultdict(set)
    
    for dish_id, actual_rating in ratings_test_dict[user]:
        actual_rating = float(actual_rating)
        if actual_rating >= 3.0:
            test_user_seen_like[user].add(dish_id)
        else:
            test_user_seen_not_like[user].add(dish_id)

    total10[user].append(get_top_n_docs(10, user, test_user_seen_like[user], test_user_seen_not_like[user]))
    avg_precision10.append(calculate_precision(10,  test_user_seen_like[user], total10[user]))
    avg_recall10.append(calculate_recall(10,  test_user_seen_like[user], total10[user]))

    total20[user].append(get_top_n_docs(20, user, test_user_seen_like[user], test_user_seen_not_like[user]))
    avg_precision20.append(calculate_precision(20,  test_user_seen_like[user], total20[user]))
    avg_recall20.append(calculate_recall(20,  test_user_seen_like[user], total20[user]))


print(f"Task 2 Precision@10: {sum(avg_precision10)/len(avg_precision10)}")
print(f"Task 2 Recall@10: {sum(avg_recall10)/len(avg_recall10)}")
print(f"Task 2 Precision@20: {sum(avg_precision20)/len(avg_precision20)}")
print(f"Task 2 Recall@20: {sum(avg_recall20)/len(avg_recall20)}")
