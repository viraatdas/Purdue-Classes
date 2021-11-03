from collections import defaultdict

# file q1.py
print("==[Q1:START]==")
import subprocess

max_degree_node = None
max_degree = -1

# this ensures that if there are duplicate edge connections listed , 
# it gets accounted by the set
adj_list = defaultdict(lambda: 0)
seen = set()


filename = "LDA_graph_edges.csv.gz"
with subprocess.Popen(['zcat',filename], stdout=subprocess.PIPE) as pipe_f:
    for line in pipe_f.stdout:
        # Must convert a byte string into an character string. Will use UTF-8 encoding.
        line = line.decode("utf-8")
        node1, node2 = [int(x) for x in line.split(",")]
       
        min_node, max_node = min(node1, node2), max(node1, node2)
        if (min_node, max_node) in seen:
            continue

        seen.add((min_node, max_node))

        adj_list[node1] += 1
        adj_list[node2] += 1

        if adj_list[node1] > max_degree:
            max_degree = adj_list[node1]
            max_degree_node = node1
        
        if adj_list[node2] > max_degree:
            max_degree = adj_list[node2]
            max_degree_node = node2
    

print(max_degree_node)

print("==[Q1:END]==")

