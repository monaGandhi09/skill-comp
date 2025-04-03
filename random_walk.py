import json
import random
from tqdm import tqdm

# Build subject adjacency list for the graph
def build_subj_adjacency_list(graph):
    adjacency_list = {node["object_id"]: [] for node in graph["objects"]}
    for edge in graph["relationships"]:
        adjacency_list[edge["subject_id"]].append((edge["predicate"], edge["object_id"]))
    return adjacency_list

# Build object adjacency list
def build_obj_adjacency_list(graph):
    adjacency_list = {node["object_id"]: [] for node in graph["objects"]}
    for edge in graph["relationships"]:
        adjacency_list[edge["object_id"]].append((edge["predicate"], edge["subject_id"]))
    return adjacency_list

# choose from a excluding items in b 
def random_choice_excluding(a, b):
    eligible_items = [item for item in a if item not in b]
    
    if not eligible_items:
        return None  
    return random.choice(eligible_items)

# get a random node from the graph which has not been walked yet
def get_random_node(graph, used_nodes, seen_nodes):
    current_node = random_choice_excluding(graph['objects'], seen_nodes)['object_id']
    while current_node in used_nodes:
        if len(used_nodes) == len(graph['objects']):
            return None
        current_node = random_choice_excluding(graph['objects'], seen_nodes)['object_id']
    return current_node

def get_neighbors(node, subj_adj, obj_adj, walk_edges, neighbors):
    
    subj_neighbors = subj_adj[node]
    for n in subj_neighbors:
        edge = {"predicate": n[0], "subject_id": node, "object_id": n[1]}
        # only add edges which are not already traversed
        if edge not in walk_edges and edge not in neighbors:
            neighbors.append(edge)
            
    obj_neighbors = obj_adj[node]
    for n in obj_neighbors:
        edge = {"predicate": n[0], "subject_id": n[1], "object_id": node}
        # only add edges which are not already traversed
        if edge not in walk_edges and edge not in neighbors:
            neighbors.append(edge)

# Random walk algorithm
def random_walk(graph, start_node, steps):
    subj_adjacency_list = build_subj_adjacency_list(graph)
    obj_adjacency_list = build_obj_adjacency_list(graph)
    current_node = start_node
    walk_nodes = {current_node}
    walk_edges = []
    seen_nodes = []
    neighbors = []
    steps_taken = 0

    for _ in range(steps):
        
        seen_nodes.append(current_node)
        # get all the neighbors from both adjacency matrices
        get_neighbors(current_node, subj_adjacency_list, obj_adjacency_list, walk_edges, neighbors)
        
        # set the #of attemps to be 10 
        attempt = 10
        
        # if the neighbors list is empty get another random node and make the neighbors list
        while not neighbors:
            current_node = get_random_node(graph, walk_nodes, seen_nodes)
            if current_node != None:
                get_neighbors(current_node, subj_adjacency_list, obj_adjacency_list, walk_edges, neighbors)
                seen_nodes.append(current_node)
            attempt -= 1
            if attempt == 0:
                break
            
        # if the neighbors list is still empty break 
        if not neighbors:
            break
        
        # else pick a random neighbor from the list 
        neighbor = random.choice(neighbors)
        
        neighbors.remove(neighbor)
        # add next node
        walk_nodes.add(neighbor['object_id'])
        # if current node not in walk nodes add it as well
        if neighbor['subject_id'] not in walk_edges:
            walk_nodes.add(neighbor['subject_id'])
        walk_edges.append(neighbor)
        steps_taken += 1
        current_node = neighbor['object_id']
        
        if len(walk_nodes) == len(graph['objects']):
            break

    return walk_nodes, walk_edges, steps_taken

# Extract subgraph based on the walk
def extract_subgraph(graph, walk_nodes, walk_edges):
    # Filter nodes
    subgraph_nodes = {node["object_id"]:node for node in graph["objects"] if node["object_id"] in walk_nodes}
    # Use the walk_edges directly for the subgraph edges
    subgraph_edges = walk_edges

    return {
        "graph":graph,
        "objects": subgraph_nodes,
        "relations": subgraph_edges,
    }

file_path = "scene_graphs.json"
save_path = "subgraphs_dict"

with open(file_path, "r") as f:
    data = json.load(f)
    
steps = 10  # Number of steps in the random walk

for graph in tqdm(data):
    id = graph['image_id']
    
    if len(graph['objects']) == 0:
        continue
    
    start_node = random.choice(graph['objects'])['object_id'] 

    walk_nodes, walk_edges, steps_taken = random_walk(graph, start_node, steps)

    subgraph = extract_subgraph(graph, walk_nodes, walk_edges)
    
    with open(f"{save_path}/{id}.json", "w") as f:
        json.dump(subgraph, f, indent=4)
    

    
