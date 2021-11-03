import time
import numpy as np

from pyspark import SparkContext, SparkConf
from pyspark.sql import *
from pyspark.sql.types import *


def load_data(spark_context):
    """Load the data into Spark as SQL tables

    The tables are:
        - Movies(MovieID, Title, Genres)
        - Users(UserID, Gender, Age, Occupation, Zipcode)
        - Ratings(UserID, MovieID, Rating, Timestamp)
    """
    sql_context = SQLContext(spark_context)

    # table movies;
    input_file = f"/user/cs49000/data/hw4/movies.dat"
    print(f".. Reading data from {input_file}")
    lines = spark_context.textFile(input_file)
    parts = lines.map(lambda l: l.split("\t"))
    parts_stripped = parts.map(lambda p: (int(p[0]), p[1], p[2].strip()))
    parts_with_index = parts_stripped.zipWithIndex().map(lambda x: x[0] + (x[1],))
    fields = [
        StructField("MovieID", IntegerType(), True),
        StructField("Title", StringType(), True),
        StructField("Genres", StringType(), True),
        StructField("MovieIndex", IntegerType(), True),
    ]
    schema = StructType(fields)
    schemaDF = sql_context.createDataFrame(parts_with_index, schema)
    schemaDF.registerTempTable("Movies")

    # Load pre-trained movies factors
    V_txt = spark_context.textFile("/user/cs49000/data/hw4/movies_factors")
    V_np = V_txt.map(lambda line: line.split(","))
    # Create an RDD with (MovieID, latent_factor)
    V_np = V_np.map(lambda row: (int(row[0]), np.array([float(x) for x in row[1:]])))
    # Join with a mapping (MovieID, MovieIndex) -> (MovieID, (latent_factor, MovieIndex))
    V_pair = V_np.join(parts_with_index.map(lambda p: (p[0], p[-1])))
    # Strip MovieID, sort by index, collect and convert to matrix
    V = np.array(V_pair.map(lambda x: (x[1][1], x[1][0])).sortByKey().values().collect())

    # table ratings;
    input_file = f"/user/cs49000/data/hw4/ratings.dat"
    print(f".. Reading data from {input_file}")
    lines = spark_context.textFile(input_file)
    parts = lines.map(lambda l: l.split("\t"))
    parts_stripped = parts.map(lambda p: (int(p[0]), int(p[1]), float(p[2]), p[3].strip()))
    fields = [
        StructField("UserID", IntegerType(), True),
        StructField("MovieID", IntegerType(), True),
        StructField("Rating", FloatType(), True),
        StructField("Timestamp", StringType(), True),
    ]
    schema = StructType(fields)
    schemaDF = sql_context.createDataFrame(parts_stripped, schema)
    schemaDF.registerTempTable("Ratings")

    # table users;
    input_file = f"/user/cs49000/data/hw4/users.dat"
    print(f".. Reading data from {input_file}")
    lines = spark_context.textFile(input_file)
    parts = lines.map(lambda l: l.split("\t"))
    parts_stripped = parts.map(lambda p: (int(p[0]), p[1], int(p[2]), p[3], p[4].strip()))
    parts_with_index = parts_stripped.zipWithIndex().map(lambda x: x[0] + (x[1],))
    fields = [
        StructField("UserId", IntegerType(), True),
        StructField("Gender", StringType(), True),
        StructField("Age", IntegerType(), True),
        StructField("Occupation", StringType(), True),
        StructField("Zipcode", StringType(), True),
        StructField("UserIndex", IntegerType(), True),
    ]
    schema = StructType(fields)
    schemaDF = sql_context.createDataFrame(parts_with_index, schema)
    schemaDF.registerTempTable("Users")

    # Load pre-trained users factors
    U_txt = spark_context.textFile("/user/cs49000/data/hw4/users_factors")
    U_np = U_txt.map(lambda line: line.split(","))
    # Create an RDD with (UserID, latent_factor)
    U_np = U_np.map(lambda row: (int(row[0]), np.array([float(x) for x in row[1:]])))
    # Join with a mapping (UserID, UserIndex) -> (UserID, (latent_factor, UserIndex))
    U_pair = U_np.join(parts_with_index.map(lambda p: (p[0], p[-1])))
    # Strip UserID, sort by index, collect and convert to matrix
    U = np.array(U_pair.map(lambda x: (x[1][1], x[1][0])).sortByKey().values().collect())

    # Since the IDs of the users and movies might not match their sequential
    # index, we added their contiguous indices as an extra column. Now, we
    # Perform a join of the three tables, to get triplets of the form
    # (user, movie, rating), where `user` and `movie` are indices instead
    query = """
    SELECT UserIndex, MovieIndex, Rating
    FROM Ratings R
        JOIN Users U ON R.UserID = U.UserID
        JOIN Movies M on R.MovieID = M.MovieID
    """
    ratings = sql_context.sql(query).coalesce(2).cache()

    return ratings, (U, V)


def train(spark_context, ratings, pre_trained_factors=None, num_latent_factors=10):
    """Train a Collaborative Filtering model using gradient descent.

    The data is stored in Spark as SQL tables.

    Args:
        spark_context (SparkContext): the Spark context for current
            spark session.
        ratings (DataFrame): the data used to train the model
        pre_trained_factors (tuple of np.ndarray): pre-trained factors
        num_latent_factors (int): number of latent factors to use

    Returns:
        user_factor (np.ndarray): User latent factors
        movie_factor (np.ndarray): Movies latent factor
    """

    # Create a SQL context to carry out SQL operation in Spark
    sql_context = SQLContext(spark_context)

    # Initialize model
    print(f" .. Initializing model with random factors")
    if pre_trained_factors is not None:
        users_factors, movies_factors = pre_trained_factors
    else:
        users_factors, movies_factors = initialize_factors(sql_context, num_latent_factors)

    # Parameters of the optimization
    learning_rate = 1
    regularization_strength = 1
    max_epochs = 30

    print(" .. Starting Gradient Descent loop")
    for epoch in range(max_epochs):
        start_time = time.time()

        # Compute the gradients
        # Broadcast phase
        factors = (spark_context.broadcast(users_factors), spark_context.broadcast(movies_factors))
        # Map and Reduce phase
        grad_users, grad_movies = compute_gradients(ratings, factors, regularization_strength)

        # Update the parameters
        users_factors = users_factors - learning_rate * grad_users
        movies_factors = movies_factors - learning_rate * grad_movies

        mse = evaluate(ratings, (users_factors, movies_factors))

        elapsed_time = time.time() - start_time
        print(f"[Epoch {epoch}] LR: {learning_rate:.2e}, MSE: {mse:.3f} [Time: {elapsed_time:.2f}s]")

    return users_factors, movies_factors


def initialize_factors(sql_context, n_factors):
    # Count number of users and movies
    query = "SELECT COUNT(DISTINCT UserID) as n_users FROM Users"
    n_users = sql_context.sql(query).first().n_users
    query = "SELECT COUNT(DISTINCT MovieID) as n_movies FROM Movies"
    n_movies = sql_context.sql(query).first().n_movies

    # Create the initial random factors
    users_factor = np.random.rand(n_users, n_factors)
    movies_factor = np.random.rand(n_movies, n_factors)

    return users_factor, movies_factor


def compute_gradients(ratings, factors, _lambda):
    """Compute gradients for the given ratings

    Args:
        ratings (pyspark.DataFrame): tuples of (user_id, movie_id, rating)
        factors (tuple of np.ndarray): users and movies factors
        _lambda (float): regularization strength
    """
    users_factors, movies_factors = factors

    def per_obs_gradient(x):
        """This will take a rating (u, i, r_ui) and compute its contribution
        to the gradient of the latent factors of U_u and V_i

        Returns:
            (u, grad_u), (v, grad_v)
        """
        # Input: triplet (user_idx, movie_idx, rating)
        u, i, r_ui = x
        U_u = users_factors.value[u]
        V_i = movies_factors.value[i]
        tmp = -(r_ui - np.dot(U_u, V_i))
        return (u, tmp * V_i), (i, tmp * U_u)

    # Compute the contribution of each rating to the gradients
    per_rating_gradients = ratings.rdd.map(per_obs_gradient)

    # Separate the contributions for users and movies into two separate RDDs
    # Each item in the RDD for the users will have a tuple
    # (u, grad_Uu_from_that_rating), and the RDD for the movies will have a tuple
    # (i, grad_Vi_from_that_rating).
    gradient_users_rdd = per_rating_gradients.map(lambda x: x[0])
    gradient_movies_rdd = per_rating_gradients.map(lambda x: x[1])

    # Compute the gradients for the users latent factors, the variable
    # `gradient_users_rdd` holds, for each rating in the dataset, its
    # contribution to the gradient of the latent factor of that user.
    # The following reduce operation will combine, for each user, the
    # contributions from all ratings from that user.
    gradient_users_rdd = gradient_users_rdd.reduceByKey(lambda a, b: b + a)
    grad_users = np.zeros_like(users_factors.value)
    for u, grad_u in gradient_users_rdd.collect():
        grad_users[u] = (_lambda * users_factors.value[u]) + grad_u

    # Compute the gradients for the movies latent factors, the variable
    # `gradient_movies_rdd` holds, for each rating in the dataset, its
    # contribution to the gradient of the latent factor of that movie.
    # The following reduce operation will combine, for each movie, the
    # contributions from all ratings for that movie.
    gradient_movies_rdd = gradient_movies_rdd.reduceByKey(lambda a, b: b + a)
    grad_movies = np.zeros_like(movies_factors.value)
    for i, grad_v in gradient_movies_rdd.collect():
        grad_movies[i] = (_lambda * movies_factors.value[i]) + grad_v

    # Re-normalize the gradients
    norm = np.sqrt((grad_users ** 2).sum() + (grad_movies ** 2).sum())
    grad_users /= norm
    grad_movies /= norm

    return grad_users, grad_movies


def evaluate(ratings, factors):
    """Compute MSE between actual ratings and predicted ratings"""
    users_factors, movies_factors = factors

    def square_error(x):
        """Transforms a tuple (UserID, MovieID, True Rating) into the
        squared error: (r_ui - dot(U_u, V_i))^2 """
        user, movie, true_rating = x

        # Compute predicted rating as the dot product
        pred_rating = np.dot(users_factors[user], movies_factors[movie])

        return (true_rating - pred_rating) ** 2

    # Transform the tuples of ratings into square error and compute mean
    mse = ratings.rdd.map(square_error).mean()

    return mse


def recommend(spark_context, latent_factors):
    U, V = latent_factors

    sql_ctx = SQLContext(spark_context)

    # Run a query to get all pairs of (Users, Movies) which were not rated
    query = """
    SELECT
      U.UserIndex,
      M.MovieIndex
    FROM
      Users U
      CROSS JOIN Movies M
    WHERE
      M.MovieID NOT IN (
        SELECT MovieID
        FROM Ratings
        WHERE Ratings.UserID = U.UserID
      )
    """
    not_rated = sql_ctx.sql(query)

    # This will compute ratings according to model
    def predict(t):
        Uu = U[t.UserIndex]
        Vi = V[t.MovieIndex]
        r = float(np.dot(Uu, Vi ))
        return t.UserIndex, (t.MovieIndex, r)

    # Predict ratings for the pairs (user, movie) which don't have
    # ratings yet in the dataset
    predicted_ratings = not_rated.rdd.map(predict)

    # Use a reduce operation to select, for each user, the movie with largest
    # predicted rating.
    # This function will be called with two pairs of movies:
    # pair_1 = (Movie1, Rating1) and pair_2 = (Movie2, Rating2), for the same
    # user. You should return the pair with the highest rating
    def reduce_max(pair_1, pair_2):
        return max(pair_1, pair_2, key=lambda x: x[1])
    top_prediction = predicted_ratings.reduceByKey(reduce_max)

    # Transform into DataFrame
    rows = top_prediction.map(lambda x: Row(UserIndex=x[0], MovieIndex=x[1][0], Rating=x[1][1]))
    recommendations = sql_ctx.createDataFrame(rows)

    # Convert Index to ID and get movie titles
    users = sql_ctx.sql("SELECT UserID, UserIndex FROM Users")
    movies = sql_ctx.sql("SELECT MovieIndex, Title FROM Movies")
    recommendations = recommendations.join(users, on="UserIndex").join(movies, on="MovieIndex")
    recommendations = recommendations.select(["UserID", "Title", "Rating"])

    return recommendations


# ==================================================== #


if __name__ == '__main__':
    print(f"===[Q1 Start]===")

    np.random.seed(42)

    conf = SparkConf().setAppName("HW-4, Q1")
    spark_context = SparkContext(conf=conf)

    # Load the data (as a spark RDD)
    print(">> Loading data.")
    all_ratings, pre_trained_factors = load_data(spark_context)

    # Train the collaborative filtering with gradient descent
    print(">> Learning latent factors for collaborative filtering ")
    latent_factors = train(spark_context, all_ratings, pre_trained_factors, num_latent_factors=10)

    print(">> Recommendations:")
    recommended = recommend(spark_context, latent_factors)
    for row in recommended.orderBy("UserID", ascending=True).collect():
        print(f"{row.UserID:5} (Rating: {row.Rating:5.2f}) =>", row.Title)

    print(f"===[Q1 End]===")
