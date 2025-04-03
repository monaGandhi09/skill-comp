import json, os
import transformers
import torch
from tqdm import tqdm
import random
import copy

import argparse

def get_obj_attr(obj_id, objs, count, freeze=False):
    obj = objs[f"{obj_id}"]
    name = obj["names"][0]
    attrs = []
    if "attributes" in obj:
        attrs = [obj["attributes"][0]]
    if freeze:
        return attrs, name, count
    num = len(attrs)
    # print(count)
    if count > num:
        count -= num
        return attrs, name, count
    elif count == 0:
        return [], name, count
    else:
        # print(attrs, count)
        attrs = random.sample(attrs, count)
        return attrs, name, 0
    
def get_caption_w_complexity(data, complexity=4):
    captions = []

    relations = copy.deepcopy(data['relations'])
    objects = copy.deepcopy(data['objects']) # this is a dict 
    ids = list(data["objects"].keys())
    
    stop = 0

    while complexity > 0:
        if stop > 12:
            print("LOOP??")
            
        stop += 1

        if complexity >= 3 and len(relations) > 0:
            # get a random relation 
            rel = random.choice(relations)
            
            s_id = f"{rel['subject_id']}"
            o_id = f"{rel['object_id']}"
            
            # don't repeat with the same subject object and different relation. 
            if s_id not in ids and o_id not in ids:
                relations.remove(rel)
                continue
            
            # only then consider the relation valid and hence reduce the count by 1
            complexity -= 1
            
            if s_id in ids:
                complexity -= 1
                ids.remove(s_id)
                s_attr, s_name, complexity = get_obj_attr(rel['subject_id'], objects, complexity-1)
                complexity += 1
            else:
                s_attr, s_name, complexity = get_obj_attr(rel['subject_id'], objects, complexity, freeze=True)
            if o_id in ids:
                complexity -= 1
                ids.remove(o_id)
                o_attr, o_name, complexity = get_obj_attr(rel['object_id'], objects, complexity)
            else:
                o_attr, o_name, complexity = get_obj_attr(rel['object_id'], objects, complexity, freeze=True)

            caption = {
                "rel": rel['predicate'].lower(),
                "subj": s_name,
                "subj_id": rel['subject_id'],
                "subj_attr": s_attr,
                "obj": o_name,
                "obj_id":rel['object_id'],
                "obj_attr": o_attr
            }
            relations.remove(rel)
            captions.append(caption)
        
        else:
            # just get a random object 
            if len(ids) > 0:
                o_id = random.choice(ids)
                complexity -= 1
                ids.remove(o_id)
                o_attr, o_name, complexity = get_obj_attr(o_id, objects, complexity)
                caption = {
                    "rel": "",
                    "subj": "",
                    "subj_id": "",
                    "subj_attr": [],
                    "obj": o_name,
                    "obj_id":o_id,
                    "obj_attr": o_attr
                }
                captions.append(caption)
            else:
                return {}
            
    return captions
    
def get_all_captions(file, input_dir, output_dir, max_complexity):
    # create caption with certain number of tokens -- atleast 2 objects and 1 relation
    file_path = f"{input_dir}/{file}"
    with open(file_path, "r") as f:
        data = json.load(f)
        
    complexity_caption = {}

    for complexity in range(3, max_complexity+1):
        captions = get_caption_w_complexity(data, complexity=complexity)
        complexity_caption[complexity] = captions

    with open(f"{output_dir}/{file}", "w") as f:
        json.dump(complexity_caption, f, indent=4)


# Set up argument parser
parser = argparse.ArgumentParser(description="Process input and output files.")
parser.add_argument('-i', '--input', type=str, help="start", required=True)
parser.add_argument('-o', '--output', type=str, help="end", required=True)
parser.add_argument('-c', '--max-complexity', type=str, help="end", default=12)

# Parse arguments
args = parser.parse_args()

files = os.listdir(args.input)
max_complexity = args.max_complexity

for file in tqdm(files):
    get_all_captions(file, args.input, args.output, max_complexity)
    






