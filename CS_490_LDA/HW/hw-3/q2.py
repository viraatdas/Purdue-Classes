import time

import numpy as np

from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import functools


def load_data(spark_context, split):
    # Read the data
    input_file = f"/user/cs49000/data/hw3/{split}"
    print(f".. Reading data from {input_file}")
    data_frame = SparkSession(spark_context).read.parquet(input_file)

    # Convert DataFrame rows to tuples
    examples = data_frame.rdd.map(lambda r: (r.y, r.X.values.reshape(-1, 1)))

    # Cache the data, to avoid re-computing the transformations above
    examples = examples.cache()

    return examples


def compute_gradients(examples, params):
    """Use Spark to compute the full gradients for the logistic regression"""

    # Compute the probabilities assigned by the logistic regression to each
    # class, for each example
    probs = predict_logistic(examples, params)

    # Compute the gradients for each example using a map operation
    def per_example_gradient(values):
        probs, (y, X) = values
    
        # <your code here. Begin>
        grad_W = X * (y - probs[1])
        grad_b = y - probs[1]
        # <your code here. End>
        return (grad_W, grad_b)

    # <your code here. Begin>
    all_gradients = probs.zip(examples).map(per_example_gradient)
    # <your code here. End>

    # Use a reduce operation to sum the gradients of all examples
    def reduce_gradients(grad1, grad2):
        # <your code here. Begin>
        gW = grad1[0] + grad2[0]
        gb = grad1[1] + grad2[1]

        # <your code here. End>
        return gW, gb
    # <your code here. Begin>

    grad_W, grad_b = all_gradients.reduce(reduce_gradients)
    # <your code here. End>

    # Rescale the gradient, to avoid running into numerical errors
    norm = np.sqrt((grad_W ** 2).sum() + (grad_b ** 2))
    grad_W /= norm
    grad_b /= norm

    return grad_W, grad_b


def predict_logistic(examples, params):
    """For each example, compute the probability of belong to each of
    the two classes

    Args:
        examples: Spark RDD containing the tuples (y, X) for each example
        params: tuple (W, b) containing the logistic regression parameters

    Returns:
        probs: Spark RDD containing, for each example, a numpy array with the
            probabilities of belonging to each class. Each element of the RDD (on each server) is
            a vector of dimension 2
    """
    W, b = params

    # Use spark `map` function to compute the probabilities of belonging
    # to class 1, for each example
    # <your code here. Begin>
    p1 = examples.map(lambda v: sigmoid(v[1].T @ W + b).item())
    # <your code here. End>

    # Use spark `map` function to create an RDD with the probabilities
    # of belonging to each class (as a numpy array)
    # <your code here. Begin>
    probs = p1.map(lambda p: np.array([1-p, p]))
    # <your code here. End>

    return probs


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def compute_metrics(examples, probabilities):
    """Compute the accuracy and loss (avg. negative log-likelihood)"""

    pred_labels = probabilities.map(lambda p: int(p.argmax()))

    def is_correct(pair):
        (true_y, X), pred_y = pair
        return int(true_y == pred_y)

    def per_example_loss(pair):
        (true_y, X), probs = pair
        return -np.log(probs[true_y])

    accuracy = examples.zip(pred_labels).map(is_correct).mean() * 100
    loss = examples.zip(probabilities).map(per_example_loss).mean()

    return accuracy, loss


# ==================================================== #

if __name__ == '__main__':
    print(f"===[Q2 Start]===")

    conf = SparkConf().setAppName("HW-3, Q2")
    spark_context = SparkContext(conf=conf)

    # Load the data (as a spark RDD)
    print(">> Loading data.")
    data_train = load_data(spark_context, split="train")
    data_test = load_data(spark_context, split="test")

    # Compute dimension of the data
    d = len(data_train.first()[1])
    print(f">> Data has {d} features")

    print(f">> Train data has {data_train.count()} examples.")
    print(f">> Test data has {data_test.count()} examples.")

    # Initial parameters
    np.random.seed(42)
    W = np.random.randn(d, 1)
    b = np.random.randn(1)

    # Run Gradient Ascent
    num_epochs = 300
    lr = 1e-1
    print(f">> Beginning gradient ascent for {num_epochs} epochs")
    epoch_duration = 0
    epoch_count = 0
    for epoch in range(num_epochs):
        timer = time.time()

        # Compute gradients
        grad_W, grad_b = compute_gradients(data_train, (W, b))

        # Take a gradient step
        W = W + lr * grad_W
        b = b + lr * grad_b
        elapsed = time.time() - timer
        epoch_duration += elapsed
        epoch_count += 1

        # Evaluate model on the training data on every 10 epochs
        if epoch % 10 == 0:

            # Evaluate the model on both train and test data
            timer = time.time()
            probs_train = predict_logistic(data_train, (W, b))
            acc_train, loss_train = compute_metrics(data_train, probs_train)
            probs_test = predict_logistic(data_test, (W, b))
            acc_test, loss_test = compute_metrics(data_test, probs_test)
            elapsed_eval = time.time() - timer

            print(f"[Epoch {epoch + 1:5d} / {num_epochs:5d}] Train Accuracy: {acc_train:.2f} % "
                  f"| Test Accuracy: {acc_test:.2f} % "
                  f"| Train Avg. Loss: {loss_train:.5f} "
                  f"| Test Avg. Loss: {loss_test:.5f} ")

            epoch_duration /= epoch_count
            epoch_count = 0
            print(f"[Epoch {epoch + 1:5d} / {num_epochs:5d}] "
                  f"Time: training = {epoch_duration:.1f} s/epoch "
                  f"| evaluation = {elapsed_eval:.1f}s")

    # Print the output
    print(">> Finish training. Coeficients:")
    for i, wi in enumerate(W.flatten()):
        print(f".. w_{i} = {wi:.3f}")
    print(f".. bias = {b.item():.3f}")

    # Evaluate the final model on both train and test data
    timer = time.time()
    probs_train = predict_logistic(data_train, (W, b))
    acc_train, loss_train = compute_metrics(data_train, probs_train)
    probs_test = predict_logistic(data_test, (W, b))
    acc_test, loss_test = compute_metrics(data_test, probs_test)
    elapsed_eval = time.time() - timer

    print(f">> Model evaluation:")
    print(f"[Training data] Accuracy: {acc_train:.2f} | Avg. Loss: {loss_train:.5f}")
    print(f"[Test data] Accuracy: {acc_test:.2f} | Avg. Loss: {loss_test:.5f}")
    print(f"===[Q2 End]===")