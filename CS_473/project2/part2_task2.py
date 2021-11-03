import pandas as pd
import json
import numpy as np
from collections import defaultdict, Counter
from operator import add
from scipy.spatial import distance


dishes_df = pd.read_csv("/homes/cs473/project2/data/dishes.csv")
with open("/homes/cs473/project2/data/user_ratings_train.json") as rating_train_file:
    ratings_train_dict = json.load(rating_train_file)

with open("/homes/cs473/project2/data/user_ratings_test.json") as rating_test_file:
    ratings_test_dict = json.load(rating_test_file)

user_dishes_rating_list = [[]] * len(ratings_train_dict)

user_dish_rating_by_dish = [[]] * len(ratings_train_dict)

num_dishes = len(dishes_df)

for user in ratings_train_dict:
    dish_list = [-1] * num_dishes
    ingredient_list = [-1] * (len(dishes_df.columns) - 2)
    sum_rating = 0
    num_rating = 0

    counter_index_elements = Counter()
    for dish_id, actual_rating in ratings_train_dict[user]:
        dish_id = int(dish_id)
        

        curr_ingredients = dishes_df.loc[[dish_id]].to_numpy()
        # ignore the dish id and dishname
        curr_ingredients = curr_ingredients[0][2:]

        indices_with_elements = np.where(curr_ingredients == 1)[0]
        actual_rating = float(actual_rating)
        dish_list[dish_id] = actual_rating
        for index in indices_with_elements:
            counter_index_elements[index] += 1
            ingredient_list[index] += actual_rating
        
        for index in counter_index_elements:
            ingredient_list[index] /= counter_index_elements[index]

        sum_rating += actual_rating
        num_rating += 1
    avg = sum_rating/num_rating

    

    for i in range(len(dishes_df.columns) - 2):
        if ingredient_list[i] != -1:
            ingredient_list[i] = ingredient_list[i] - avg
            
    ingredient_list.append(avg)
    dish_list.append(avg)
    user_dishes_rating_list[int(user)] = ingredient_list
    user_dish_rating_by_dish[int(user)] = dish_list

col_names = dishes_df.columns.tolist()[2:] + ["user_avg"]
sub_mean_user_dishes_rating_df = pd.DataFrame(user_dishes_rating_list, columns=col_names)

col_names_by_dish = [f'{i}' for i in range(num_dishes)] + ["user_avg"]
user_rating_by_dish_df = pd.DataFrame(user_dish_rating_by_dish, columns=col_names_by_dish)

import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from sklearn.cluster import KMeans

def cluster_user_by_ingredients():
    X = sub_mean_user_dishes_rating_df.values
    X[X==-1] = 0
    user_labels = np.array([i for i in range(len(ratings_train_dict))])

    kmeans = KMeans(n_clusters=10).fit(X)
    pred_classes = kmeans.predict(X)

    cluster_to_user_map = {}
    for cluster in range(10):
        cluster_to_user_map[cluster] = user_labels[np.where(pred_classes==cluster)]
    return cluster_to_user_map
    

similiarity_between_users = defaultdict(lambda: [0] * len(ratings_train_dict))

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
# with open('similarity_between_users_part2.pkl', 'wb+') as file:
#     dill.dump(similiarity_between_users, file)

with open('similarity_between_users_part2.pkl', 'rb') as handle:
    similiarity_between_users = dill.load(handle)

cluster_to_user_map = cluster_user_by_ingredients()

def predict(user, dish_id):
    user = int(user)
    user_avg = sub_mean_user_dishes_rating_df.loc[int(user)]['user_avg']
    
    curr_ingredients = dishes_df.loc[[dish_id]].to_numpy()
    curr_ingredients = curr_ingredients[0][2:]

    indices_with_elements = np.where(curr_ingredients == 1)[0]
    
    for cluster in cluster_to_user_map:
        if user in cluster_to_user_map[cluster]:
            cluster_set_for_user = cluster_to_user_map[cluster]
            break
    
    rating_sum = 0
    num_el = 0
    for curr_user in cluster_set_for_user:
        if curr_user == user:
            continue
        # don't predict based on users who didn't rate the dish
        if user_rating_by_dish_df.iloc[curr_user, dish_id] == -1:
            continue

        for curr_dish, rating in ratings_train_dict[str(curr_user)]:
            if curr_dish == dish_id:
                rating_sum += rating
                num_el+=1
                break
        
    if num_el == 0:
        return 5/2
    return rating_sum/num_el
    

def get_top_n_docs(n, user, like_set, not_like_set):
    user = int(user)
    # list of cosine similarity scores
    vals = similiarity_between_users[user]

    dishes_user_not_rated = set()

    for dish_id, val in enumerate(user_rating_by_dish_df.values[user]):
        if val == -1:
            dishes_user_not_rated.add(dish_id)
    # these are values that we should predict
    # if a dish doesn't occur here that means ignore the values
    seen_test_set = like_set | not_like_set

    for cluster in cluster_to_user_map:
        if user in cluster_to_user_map[cluster]:
            cluster_set_for_user = cluster_to_user_map[cluster]
            break
    
    # cosine_scores = []
    # for user_new in cluster_set_for_user:
    #     cosine_scores.append(vals[user_new])

    top_dishes = []

    for user_new in cluster_set_for_user:
        curr_user_values = user_rating_by_dish_df.values[int(user_new)]
        for dish_id in np.argsort(curr_user_values)[::-1]:

            if curr_user_values[dish_id] == -1 or dish_id not in dishes_user_not_rated:
                continue
            if dish_id in seen_test_set:
                top_dishes.append(dish_id)
        
            if len(top_dishes) >= n:
                break
    if len(top_dishes) == 0:
        cosine_score_index = {score: i for i, score in enumerate(vals)}
        top_dishes = []
        
        for score, index in sorted(cosine_score_index.items(), reverse=False):
            curr_user_values = user_rating_by_dish_df.values[index]
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


# # def get_row_as_list_from_df(df, column_name, column_value):
# #     return df.loc[df[column_name] == column_value].values.tolist()

# # dishid_to_users_dict = defaultdict(list)
# # for user, dish_and_rating in ratings_train_dict.items():
# #     dish, rating = dish_and_rating
# #     dishid_to_users_dict[dish].append((user, rating))


# # for user_id in ratings_test_dict:
# #     for dish_id, actual_rating in ratings_test_dict[user_id]:
# #         for user_id, user_rating in dishid_to_users_dict[dish_id]:



# # # prepare data
# # new_cols = dishes_df.columns.tolist()
# # new_cols[0] = 'user_id'
# # train_dict = {}

# # for user_id in ratings_train_dict:
# #     for dish_id, rating in ratings_train_dict[user_id]:

# #         # if row in df with particular user_id doesn't exist
# #         if user_id not in train_dict:
# #             curr_dish_list = get_row_as_list_from_df(dishes_df, 'dish_id', dish_id)
# #             curr_dish_list = curr_dish_list[0][2:]
# #             train_dict[user_id] = [i * rating for i in  curr_dish_list]
# #         else:
# #             curr_dish_list = get_row_as_list_from_df(dishes_df, 'dish_id', dish_id)
# #             curr_dish_list = curr_dish_list[0][2:]
# #             curr_train_user_list = train_dict[user_id]

# #             new_list = [i * rating for i in  curr_dish_list]
# #             new_list = list(map(add, new_list, curr_train_user_list))

# #             for i, el in enumerate(curr_dish_list):
# #                 if el != 0:
# #                     new_list[i] /=2

# #             train_dict[user_id] = new_list    

# # # get predicted values and MAE 
# # def get_predicted_rating(user_id, ingredient_list):
# #     curr_train_user_list = train_dict[user_id]
# #     curr_sum = 0
# #     num_el = 0
# #     for i, el in enumerate(ingredient_list):
# #         if el != 0:
# #             num_el += 1
# #             curr_sum += curr_train_user_list[i]
# #     if num_el == 0:
# #         return 2.5
# #     return curr_sum/num_el
    

# # mae = {}
# # sum_err = 0
# # num_el = 0
# # for user_id in ratings_test_dict:
# #     for dish_id, actual_rating in ratings_test_dict[user_id]:
# #         curr_dish_list = get_row_as_list_from_df(dishes_df, 'dish_id', dish_id)
# #         curr_dish_list = curr_dish_list[0][2:]

# #         predicted_rating = get_predicted_rating(user_id, curr_dish_list)
        
# #         sum_err += abs(predicted_rating - actual_rating)
# #         num_el += 1

# # print(sum_err/num_el)