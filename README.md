## Step 1: Creating a subgraph to work with
Run: ``python random_walk.py``
input: scene_graphs.json
output: file for each image in subgraphs_dict/
Each file contains the subgraph from the scene to create captions

## Step 2: Getting captions with different complexities
Run: ``python get_complexity.py --input ${input directory} --output ${output directory} --max_complexity ${max complexity}``
input directory: subgraphs_dict
output directory: subgraphs_caption
max complexity: 12 (default)

## Step 3: Getting hard negatives 