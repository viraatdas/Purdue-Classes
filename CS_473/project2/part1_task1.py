import pandas as pd
import json
import numpy as np
from collections import defaultdict
from operator import add
from scipy.spatial import distance





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



with open("/homes/cs473/project2/data/user_ratings_test.json") as rating_test_file:
    ratings_test_dict = json.load(rating_test_file)

import warnings
warnings.filterwarnings("error")
def cos_sim(a, b, index1, index2):
  
    a = np.asarray(a)
    b = np.asarray(b)

    a_0 = np.where(a == -1)[0]
    b_0 = np.where(b == -1)[0]
    
    combined_set_indices = set()
    for el in a_0:
        combined_set_indices.add(el)
    for el in b_0:
        combined_set_indices.add(el)
    
    a = np.delete(a, list(combined_set_indices), axis=0)
    b = np.delete(b, list(combined_set_indices), axis=0)
    
    try:
        return distance.cosine(a, b)
    except:
        return 0
    

similiarity_between_users = defaultdict(lambda: [0] * len(ratings_train_dict))

def fill_similarity_table():
    for index1, a in sub_mean_user_dishes_rating_df.iterrows():
        a = a.tolist()[:-1]
        for index2, b in sub_mean_user_dishes_rating_df.iterrows():
            if index1 == index2:
                continue
            
            b = b.tolist()[:-1]
            sim_score = cos_sim(a, b, index1, index2)
            
            similiarity_between_users[index1][index2] = sim_score

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
    if sub_mean_user_dishes_rating_df.loc[user][dish_id] != -1:
        return sub_mean_user_dishes_rating_df.loc[user][dish_id]
  
    user_avg = sub_mean_user_dishes_rating_df.loc[int(user)]['user_avg']
    summation = 0
    sum_similarity_score = 0
    for index in range(len(sub_mean_user_dishes_rating_df)):
        if index == user:
            continue
        if sub_mean_user_dishes_rating_df.iloc[index, dish_id] == -1:
            continue

        sim_value = similiarity_between_users[user][index]
        summation += sub_mean_user_dishes_rating_df.iloc[index, dish_id] * sim_value
        sum_similarity_score += abs(sim_value)

    return user_avg + summation/sum_similarity_score
    

mae = []
for user in ratings_test_dict:
    for dish_id, actual_rating in ratings_test_dict[user]:
        actual_rating = float(actual_rating)
        predicted_rating = predict(user, dish_id)
        mae.append(abs(actual_rating - predicted_rating))


print(f"Task 1 MAE: {sum(mae)/len(mae)}")

