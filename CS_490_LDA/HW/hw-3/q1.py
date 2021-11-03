import random
import math
import collections
from functools import reduce
import multiprocessing as mp
import time

# ==================================================== #


def sigmoid(u):
    """Compute the sigmoid function"""
    return 1 / (1 + math.exp(-u))


def estimate_coin_prob_closed_form(X):
    """Estimate the parameter `p` of a biased coin based on the sequence of
    coin tosses.
    
    Args:
        X (list of int): sequence of coin tosses. Each element is either 0 if the
            coin toss resulted in tails or 1 if it resulted in heads.

    Returns:
        p (float): estimated probability of heads
    """

    # <your code here. Begin>
    p = np.sum(X)/len(X)
    # <your code here. End>

    return p


def estimate_coin_prob_gd(X, version="simple"):
    """Estimate the parameter `p` of a biased coin based on the sequence of
    coin tosses using Gradient Descent. A parameter controls whether the 
    gradients are computed serially, using single-process map-reduce or
    multi-process map-reduce.

    Args:
        X (list of int): sequence of coin tosses. Each element is either 0 if the
            coin toss resulted in tails or 1 if it resulted in heads.
        version (str): which version of the code to use for computing
            gradients. Options are:
            - "simple" : simple single-process
            - "mapreduce" : map-reduce single-process
            - "multiprocess" :  map-reduce multiprocess

    Returns:
        p (float): estimated probability of heads
    """
    version = version.lower()
    assert version in ('simple', 'mapreduce', 'multiprocess')

    # Maximum number of iterations and learning rate:
    max_iter = 1000
    lr = 1e-5

    # Initialized the parameter with some random value between 0 and 1
    random.seed(42)

    # We'll use a little trick to avoid numerical problems:
    # we substitute p = sigmoid(u)
    # now, u can take any real value
    u = random.normalvariate(0, 1)

    # If we're running the multiprocess version, create a pool of processes
    # to compute the gradients in parallel. We create it outside of the main
    # loop to avoid unnecessary system calls to create and destroy the processes
    # which would happen if we re-created the pool at each iteration
    if version == "multiprocess":
        # Cap the number of processes at 20, to avoid errors when running on
        # the server that has a limit on the number of child processes
        num_processes = min(mp.cpu_count(), 20)

        # Create the pool
        pool = mp.Pool(processes=num_processes)

    for epoch in range(max_iter):
        p = sigmoid(u)

        # Compute the gradient of the negative log-likelihood
        if version == "simple":
            grad_p = compute_grad_p(X, p)
        elif version == "mapreduce":
            grad_p = compute_grad_p_mapreduce(X, p)
        else:

            grad_p = compute_grad_p_multiprocess(X, p, pool)

        # we compute the gradient w.r.t. u by applying the chain rule:
        # grad_u = df/dp * dp/du
        # For the sigmoid function, dp/du = p * (1 - p)
        grad_u = grad_p * (p * (1 - p))

        # Take a gradient step
        u = u - lr * grad_u

    # At this point, for the multiprocess version, all tasks have finished and
    # we are done with the pool, so we close it to release unneeded resources
    if version == "multiprocess":
        pool.close()

    # Convert the final `u` to [0, 1] via the sigmoid function
    p = sigmoid(u)

    return p


def compute_grad_p_single_example(xi, p):
    """Compute the gradient of the negative log-likelihood wrt p,
    for a single example (coin toss).

    Args:
        xi (int): result of the coin toss, 0 if tail, 1 if head.
        p (float): current value for the probability of heads

    Returns:
        grad_p (float): gradient of the log-likelihood for this single example
    """
    # <your code here. Begin>    
    grad_p = -(xi * 1/p - (1-xi) * 1/(1-p)) 
    # <your code here. End>
    return grad_p


def compute_grad_p(X, p):
    """Compute the gradient of the negative log-likelihood wrt p.

    Args:
        X (list of int): sequence of coin tosses. Each element is either 0 if the
            coin toss resulted in tail or 1 if it resulted in head.
        p (float): current value for the probability of heads

    Returns:
        grad_p (float): gradient of the log-likelihood wrt the parameter p
    """
    n_heads = sum(X)
    n_tails = len(X) - n_heads

    # <your code here. Begin>
    gradients = map(lambda xi: compute_grad_p_single_example(xi, p), X)
    grad_p = reduce((lambda a, b: a+b), gradients)
    # <your code here. End>

    return grad_p


def compute_grad_p_mapreduce(X, p):
    """Compute the gradient of the negative log-likelihood wrt p.
    Uses a map-reduce framework to compute the gradient of each example
    separately.

    Args:
        X (list of int): sequence of coin tosses. Each element is either 0 if the
            coin toss resulted in tail or 1 if it resulted in head.
        p (float): current value for the probability of heads

    Returns:
        grad_p (float): gradient of the log-likelihood wrt the parameter p
    """

    # Since the map function takes only one parameter (the coin toss),
    # we must wrap the actual gradient computation with this nested
    # function. This creates a function that take a single argument,
    # but inside it we'll have access to the variable `p`
    def map_function(xi):
        return compute_grad_p_single_example(xi, p)

    # Apply this map function over each example
    grad_each_example = map(map_function, X)

    # Reduce the individual gradients using sum
    grad_p = reduce(lambda a, b: a + b, grad_each_example)

    return grad_p


def compute_grad_p_multiprocess(X, p, pool):
    """Compute the gradient of the negative log-likelihood wrt p.
    Uses a map-reduce framework to compute the gradient of each example
    separately.

    Args:
        X (list of int): sequence of coin tosses. Each element is either 0 if the
            coin toss resulted in tail or 1 if it resulted in head.
        p (float): current value for the probability of heads
        pool (multiprocessing.Pool): pool of process that we'll use to compute
            the gradients in parallel

    Returns:
        grad_p (float): gradient of the log-likelihood wrt the parameter p
    """

    # First, split the data into batches, one for each process

    # Compute size of each batch. The max handles the case when there are more
    # processes than data points
    batch_size = max(len(X) // pool._processes, 1)
    # Divide the data into the batches of size `batch_size`
    # The variable `batches` should contain a list of lists of ints
    # (i.e. a list of batches of data)
    # <your code here. Begin>
    batches = [X[i:i + batch_size] for i in range(0, len(X), batch_size) if len(X[i:i + batch_size]) == batch_size]
    # <your code here. End>

    # Start the processing of each batch of data
    # Since the `apply` function blocks until the work is done, we use the
    # asynchronous version, which we must check later until it is done
    # Note: we use collections.deque so we can process the tasks in a FIFO
    # order, without added costs of adding/removing them from a list
    running_tasks = collections.deque()
    for batch in batches:
        # If this batch is empty (happens when there are more processes than
        # data points), we do nothing
        if not batch:
            continue
        # Otherwise, we start the computation in async mode and store the task
        # AsyncResult object, to collect the result later
        else:
            result_obj = pool.apply_async(compute_grad_p, (batch, p))
            running_tasks.append(result_obj)

    # We'll store the computed gradient here
    grad_p = 0

    # Collect all results of the tasks as they get done
    while running_tasks:
        # Get one of the remaining tasks (`popleft` has O(1) cost)
        task = running_tasks.popleft()

        # Try to get the result of the task. The `get` function will wait up to
        # 5 seconds for the result of the task. If the task is not yet finished
        # within the 5 seconds, we'll get a time-out error (but the task will
        # still be running)
        try:
            grad_batch = task.get(5)
        # If we get the timeout error, then the task is not finished yet, so we
        # add it back to the end of our FIFO queue, so we check it again later
        except mp.TimeoutError:
            running_tasks.append(task)
        # If we didn't get a time-out error, then we got the result of the task,
        # which is the gradient of that batch of data
        else:
            # Accumulate the gradients of the batch (grad_batch) in 
            # the variable `grad_p`
            # <your code here. Begin>
            grad_p += grad_batch
            # <your code here. End>

    return grad_p

# ==================================================== #


if __name__ == "__main__":
    print("===[Q1 Start]===")

    # Simulate coin tosses
    import numpy as np
    np.random.seed(42)
    true_p = 0.265
    n = 10000
    coin_tosses = (np.random.rand(n) < true_p).astype(int).tolist()

    # ========================== #

    print("Estimating coin bias with closed-form MLE:")

    # Estimate coin probability with 10 coin tosses (closed_form)
    start_time = time.time()
    p_hat = estimate_coin_prob_closed_form(coin_tosses[:10])
    elapsed = time.time() - start_time
    print(f"[ CF:    10 Coin Tosses] True p = {true_p:5.03f} | Estimated p = {p_hat:5.03f} [Time: {elapsed:.3f}s]")

    # Estimate coin probability with 1000 coin tosses (closed-form)
    start_time = time.time()
    p_hat = estimate_coin_prob_closed_form(coin_tosses[:1000])
    elapsed = time.time() - start_time
    print(f"[ CF:  1000 Coin Tosses] True p = {true_p:5.03f} | Estimated p = {p_hat:5.03f} [Time: {elapsed:.3f}s]")

    # Estimate coin probability with 10000 coin tosses (closed-form)
    start_time = time.time()
    p_hat = estimate_coin_prob_closed_form(coin_tosses)
    elapsed = time.time() - start_time
    print(f"[ CF: 10000 Coin Tosses] True p = {true_p:5.03f} | Estimated p = {p_hat:5.03f} [Time: {elapsed:.3f}s]")
    print("")

    # -------------------------- #

    print("Estimating coin bias with Gradient Descent (simple single-process version):")

    # Estimate coin probability with 10 coin tosses (Gradient Descent)
    start_time = time.time()
    p_hat = estimate_coin_prob_gd(coin_tosses[:10], version="simple")
    elapsed = time.time() - start_time
    print(f"[ GD:    10 Coin Tosses] True p = {true_p:5.03f} | Estimated p = {p_hat:5.03f} [Time: {elapsed:.3f}s]")

    # Estimate coin probability with 1000 coin tosses (Gradient Descent)
    start_time = time.time()
    p_hat = estimate_coin_prob_gd(coin_tosses[:1000], version="simple")
    elapsed = time.time() - start_time
    print(f"[ GD:  1000 Coin Tosses] True p = {true_p:5.03f} | Estimated p = {p_hat:5.03f} [Time: {elapsed:.3f}s]")

    # Estimate coin probability with 10000 coin tosses (Gradient Descent)
    start_time = time.time()
    p_hat = estimate_coin_prob_gd(coin_tosses, version="simple")
    elapsed = time.time() - start_time
    print(f"[ GD: 10000 Coin Tosses] True p = {true_p:5.03f} | Estimated p = {p_hat:5.03f} [Time: {elapsed:.3f}s]")
    print("")

    # -------------------------- #

    print("Estimating coin bias with Gradient Descent (map-reduce single-process version):")

    # Estimate coin probability with 10 coin tosses (Gradient Descent)
    start_time = time.time()
    p_hat = estimate_coin_prob_gd(coin_tosses[:10], version="mapreduce")
    elapsed = time.time() - start_time
    print(f"[ GD:    10 Coin Tosses] True p = {true_p:5.03f} | Estimated p = {p_hat:5.03f} [Time: {elapsed:.3f}s]")

    # Estimate coin probability with 1000 coin tosses (Gradient Descent)
    start_time = time.time()
    p_hat = estimate_coin_prob_gd(coin_tosses[:1000], version="mapreduce")
    elapsed = time.time() - start_time
    print(f"[ GD:  1000 Coin Tosses] True p = {true_p:5.03f} | Estimated p = {p_hat:5.03f} [Time: {elapsed:.3f}s]")

    # Estimate coin probability with 10000 coin tosses (Gradient Descent)
    start_time = time.time()
    p_hat = estimate_coin_prob_gd(coin_tosses, version="mapreduce")
    elapsed = time.time() - start_time
    print(f"[ GD: 10000 Coin Tosses] True p = {true_p:5.03f} | Estimated p = {p_hat:5.03f} [Time: {elapsed:.3f}s]")
    print("")

    # -------------------------- #

    print("Estimating coin bias with Gradient Descent (map-reduce multiprocess version):")

    # Estimate coin probability with 10 coin tosses (GD)
    start_time = time.time()
    p_hat = estimate_coin_prob_gd(coin_tosses[:23], version="multiprocess")
    elapsed = time.time() - start_time
    print(f"[ GD:    10 Coin Tosses] True p = {true_p:5.03f} | Estimated p = {p_hat:5.03f} [Time: {elapsed:.3f}s]")

    # Estimate coin probability with 1000 coin tosses (GD)
    start_time = time.time()
    p_hat = estimate_coin_prob_gd(coin_tosses[:1000], version="multiprocess")
    elapsed = time.time() - start_time
    print(f"[ GD:  1000 Coin Tosses] True p = {true_p:5.03f} | Estimated p = {p_hat:5.03f} [Time: {elapsed:.3f}s]")

    # Estimate coin probability with 10000 coin tosses (GD)
    start_time = time.time()
    p_hat = estimate_coin_prob_gd(coin_tosses, version="multiprocess")
    elapsed = time.time() - start_time
    print(f"[ GD: 10000 Coin Tosses] True p = {true_p:5.03f} | Estimated p = {p_hat:5.03f} [Time: {elapsed:.3f}s]")

    print("===[Q1 End]===")
