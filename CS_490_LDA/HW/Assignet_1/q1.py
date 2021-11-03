# # The following two lines are needed if plotting on a server, without graphical
# # display, and they need to be before importing pyplot
# import matplotlib as mpl
# mpl.use("Agg")
# import matplotlib.pyplot as plt

# p_x, p_y = zip(*points)
# hull_x, hull_y = zip(*hull)

# fig, ax = plt.subplots()
# ax.scatter(p_x, p_y, marker='x')
# ax.plot(hull_x, hull_y, color='blue')
# ax.set_xlabel("x coordinate")
# ax.set_ylabel("y coordinate")
# ax.set_title("Object: {}".format(objname))
# fig.savefig("{}.png".format(objname))

# file q1.py
print("==[Q1:START]==")
import gzip
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import math

def split(a, b, points):
    # return points on down from of ab

    return [p for p in points if np.cross(np.subtract(p, a), np.subtract(b,a)) < 0]

def expand_hull(u, v, points):
    if not points:
        return []

    # find furthest point h on the left side of the arrow from u->v
    # by definition, h must be part of the convex hull
    h = min(points, key=lambda p: np.cross(np.subtract(p,u), np.subtract(v, u)))

    # Divide search into subproblems: 
    #  S1 = set of all points that are on the left side of the arrow from h->v
    S1 = split(h, v, points)
    #  S2 = set of all points that are on the right side of the arrow from h->v
    S2 = split(u, h, points)
    # Recursively returns the other points of the convex hull in the subproblems
    return expand_hull(h, v, S1) + [h] + expand_hull(u, h, S2)

def convex_hull_func(points):
    # sort points w.r.t. x-axis
    points = sorted(points, key=lambda p: p[0])
    # u  is left-most point (w.r.t. x-axis)
    u = points[0]
    # v  is right-most point (w.r.t. x-axis)
    v = points[-1]
    # by definition, u and v must be part of the convex hull
    
    #split points into to searches up from uv and down from uv
    up = split(u, v, points)
    down = split(v, u, points)
    
    # find points belonging to convex hull on each subset
    return [v] + expand_hull(u, v, up) + [u] + expand_hull(v, u, down) + [v]
# CREATE THE NECESSARY AUXILIARY FUNCTIONS HERE

def convex_hull_multi_core(points_2d, pool, CPU_count):
    """Code to compute convex hull using multiple cores. The input is a list of 2-tuples. 
    Each tuple contains 2 floats with the 2D (projected) coordinates of each point.
    """
    convex_hull = []

    # COMPUTE MULTI-CORE CONVEX HULL HERE
    # Reference convex hull single-core implementation
    size = int(math.ceil(float(len(points_2d)) / CPU_count))

    mapped_data = [points_2d[i * size:(i + 1) * size] for i in range(CPU_count)]

    convex_hull = pool.map(convex_hull_func, mapped_data)
    return convex_hull

def project2D(points):
    """Code to project the points into the 2D plane z=0, returning just two coordinates for each point.
    """

    return [(x,y) for x,y,_ in points]

if __name__ == "__main__":
    # The following code reads the point clouds (in 3D)
    with gzip.open("point-clouds.tsv.gz", "rt") as src:
        # Keep reading while there's still input
        while src:
            # Guard the reading with a try ... catch block to detect the end of file
            try:
                # For each object, first line has the object name and how many points it has
                objname, n_points = next(src).strip().split()
            except StopIteration: # Finished reading the file
                break

            # Read the next object
            n_points = int(n_points)
            points = []
            while len(points) < n_points:
                x, y, z = map(float, next(src).strip().split())
                points.append((x, y, z))

            # Project the points into the z=0 plane
            points = project2D(points)

            # Creates a pool of worker processes, one per CPU core

            # FIX for SLURM
            CPU_count = mp.cpu_count()
            print(f"(CPU core count = {CPU_count}) ",end='') # uncomment to help debug your code, but submit it commented
            pool = mp.Pool(processes=CPU_count)
            # Compute the convex hull
            hull = convex_hull_multi_core(points, pool, CPU_count)
            print(hull)
            # Output the convex hull
            output = f"{objname}: " + ", ".join([f"({x},{y})" for x, y in hull])
            print(output)

print("==[Q1:END]==")

