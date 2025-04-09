## Step 1: Creating a subgraph to work with
Run: ``python random_walk.py`` <br />
input: scene_graphs.json <br />
output: file for each image in subgraphs_dict/ <br />
Each file contains the subgraph from the scene to create captions 

## Step 2: Getting captions with different complexities
Run: ``python get_complexity.py --input ${input directory} --output ${output directory} --max_complexity ${max complexity}`` <br />
input directory: subgraphs_dict <br />
output directory: subgraphs_caption <br />
max complexity: 12 (default)

## Step 3: Getting hard negatives 
