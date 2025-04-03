from openai import OpenAI
import os, json
import re
import random
from tqdm import tqdm
import ast 
import argparse

from gpt_queries import gpt_caption, get_obj_negative_gpt, get_rel_negative_gpt, get_neg_attr, get_neg_rel, get_neg_obj

def get_obj(d, type="obj"):
    ''' Get the corresponding object with its attributes 
        d: scene info 
        type: "obj" or "subj" 
    '''
    obj = ""
    if d[f"{type}_attr"] != []:
        for x in d[f"{type}_attr"][:-1]:
            obj += x + ", "
        obj += d[f"{type}_attr"][-1] + " "
    obj += d[type].strip('\"').strip()
    obj += f" (id:{d[f'{type}_id']})"
    return obj
         
def get_rel_objs(cap_dicts):
    ''' Get all the relations and objects from the scene info of the subgraph
        cap_dicts: list of dictionaries each containing subject, relation and object
    '''
    relations = []
    objects = []
    for cap_dict in cap_dicts:
        # only object
        if cap_dict["rel"] == "":
            obj = get_obj(cap_dict, "obj")
            objects.append(obj.strip('\"').strip())
        
        # else it is a relation
        else:
            subj = get_obj(cap_dict, "subj")
            obj = get_obj(cap_dict, "obj")
            relation = f"{subj.strip()} {cap_dict['rel']} {obj.strip()}"
            relations.append(relation.strip('\"').strip())
    return relations, objects

def prepare_input(relations, objects):
    ''' Formats the set of relations and objects for querying gpt models
        relations: list of relations (length has to be atleast one)
        objects: list of objects (without any relations)
    '''
    input = ""
    input += "relations: "
    for rel in relations[:-1]:
        input += rel + ", "
    input += relations[-1]
    if objects != []:
        input += "\nobjects: "
        for obj in objects[:-1]:
            input += obj + ", "
        input += objects[-1]
    
    return input
  
def gt_captions(data, outfile):
    ''' Creates a file with captions of different complexities using subgraph data
        data: scene graph json
        outfile: path of the file to store captions based on the subgraph
    '''
    
    captions = {}
    for complexity, cap_dicts in data.items():
        if cap_dicts == {}:
            continue
        relations, objects = get_rel_objs(cap_dicts)
        input = prepare_input(relations, objects)
        caption = gpt_caption(input)
        captions[complexity] = caption
        
    with open(outfile, "w") as f:
        json.dump(captions, f, indent=4)


def get_all_attr(file):
    ''' Get all the attributes being used in the file, includes all attributes from the scene 
        file: Name of the file
    '''
    folder = "subgraphs_dict"
    with open(f"{folder}/{file}", "r") as f:
        data = json.load(f)
    attr_list = [x['attributes'] for x in data['graph']['objects'] if "attributes" in x]
    attrs = []
    for attr in attr_list:
        attrs.extend(attr)
    return set(attrs)
        
def get_attr_captions(file, outfolder, attr_map = {}, infolder="subgraphs_caption", gt_caption_folder="gt_captions"):   
    
    # if the output file already exists do not run the code again
    if os.path.exists(f"{outfolder}/{file}"):
        return
       
    # load the data from the subgraph captions dict
    with open(f"{infolder}/{file}", "r") as f:
        data = json.load(f)
       
    # check if we already have ground truth caption, if not query gpt
    if not os.path.exists(f"{gt_caption_folder}/{file}"):
        gt_captions(data, f"{gt_caption_folder}/{file}")
    
    # load the ground truth captions from the file
    with open(f"{gt_caption_folder}/{file}", "r") as f:
        gt_captions_data = json.load(f)
        
    # if no ground truth captions to work with just return
    if gt_captions_data == {}:
        return

    neg_captions = {} # initialize caption
    
    # get all the attributes being used in order to avoid them in hard negative captions 
    all_attr = get_all_attr(file)
        
    # get the negatives
    for complexity, cap_dicts in data.items():
        
        # get the possible attributes 
        attr = []
        for cap_dict in cap_dicts:
            attr.extend(cap_dict['obj_attr'])            
            attr.extend(cap_dict['subj_attr'])
            
        # create a weights list for all the attributes 
        weights = [1 for _ in range(len(attr))]
                
        # if there is no ground truth caption or there are no attributes in the caption just continue
        if complexity not in gt_captions_data or attr == []:
            continue
        
        gt_caption = gt_captions_data[complexity]
        negatives = []
        i = 0
        while len(negatives) < 4:
            i += 1
            
            # do max 10 iterations 
            if i > 10:
                break
            
            # choose a random attr from the list 
            replace_attr = random.choices(attr, weights)[0]
            
            # check if we already have a list of negative attributes to replace this one
            if replace_attr in attr_map:
                neg_attrs = attr_map[replace_attr]
            else:
                try:
                    neg_attrs = ast.literal_eval(get_neg_attr(replace_attr))
                except:
                    continue
                attr_map[replace_attr] = neg_attrs
                
            # don't accept any attribute present in any of the captions for this image
            neg_attrs_poss = [item for item in neg_attrs if item not in all_attr]
            
            # choose at random an attribute and replace it in ground truth caption to get a negative
            if neg_attrs_poss == []:
                continue
            neg_attr = random.choice(neg_attrs_poss)
            negative = re.sub(rf"\b{re.escape(replace_attr)}\b", neg_attr, gt_caption, count=1)
            
            # make sure it is not the same as some other negative or the ground truth caption before adding to the negative list
            if negative in negatives or negative == gt_caption:
                continue
            negatives.append(negative)
            
            # Reweight the attribute weights by reducing the weight of the used attribute
            ind = attr.index(replace_attr)
            weights[ind] /= 2
            
        neg_captions[complexity] = negatives
    
    # dump the map of complexity to negative captions   
    with open(f"{outfolder}/{file}", "w") as f:
        json.dump(neg_captions, f, indent=4)
        
def get_all_objs(file):
    ''' Get all the objects being used in the file, includes all objects from the scene 
        file: Name of the file
    '''
    folder = "subgraphs_dict"
    with open(f"{folder}/{file}", "r") as f:
        data = json.load(f)
    objects_list = [x['names'] for x in data['graph']['objects']]
    objects = []
    for obj in objects_list:
        objects.extend(obj)
    return objects
        
def get_obj_captions(file, outfolder, obj_map = {}, infolder="subgraphs_caption", gt_caption_folder="gt_captions"):   
    
    # if the output file already exists do not run the code again
    if os.path.exists(f"{outfolder}/{file}"):
        return
       
    # load the data from the subgraph captions dict
    with open(f"{infolder}/{file}", "r") as f:
        data = json.load(f)
       
    # check if we already have ground truth caption, if not query gpt
    if not os.path.exists(f"{gt_caption_folder}/{file}"):
        gt_captions(data, f"{gt_caption_folder}/{file}")
    
    # load the ground truth captions from the file
    with open(f"{gt_caption_folder}/{file}", "r") as f:
        gt_captions_data = json.load(f)
        
    # if no ground truth captions to work with just return
    if gt_captions_data == {}:
        return

    neg_captions = {} # initialize negative caption list
    
    # get all the objects being used in order to avoid them in hard negative captions 
    all_objects = get_all_objs(file)
        
    # get the negatives
    for complexity, cap_dicts in data.items():
        
        # get the possible objects 
        objs = []
        for cap_dict in cap_dicts:
            objs.append(cap_dict['obj'].strip('\"').strip())            
            if cap_dict["rel"] != "":
                objs.append(cap_dict['subj'].strip('\"').strip())
                
        # create a weights list for all the objects 
        weights = [1 for _ in range(len(objs))]
                
        # if there is no ground truth caption or there are no objects in the caption just continue
        if complexity not in gt_captions_data or objs == []:
            continue
        
        gt_caption = gt_captions_data[complexity]
        negatives = []
        i = 0
        while len(negatives) < 4:
            i += 1
            
            # do max 10 iterations 
            if i > 10:
                break
            
            # choose a random object from the list 
            replace_obj = random.choices(objs, weights)[0]
            
            # check if we already have a list of negative objects to replace this one
            if replace_obj in obj_map:
                neg_objs = obj_map[replace_obj]
            else:
                try:
                    neg_objs = ast.literal_eval(get_neg_obj(replace_obj))
                except:
                    continue
                obj_map[replace_obj] = neg_objs
                
            # don't accept any object present in any of the captions for this image
            neg_objs_poss = [item for item in neg_objs if item not in all_objects]
            
            # choose at random an attribute and replace it in ground truth caption to get a negative
            if neg_objs_poss == []:
                continue
            neg_obj = random.choice(neg_objs_poss)
            negative = re.sub(rf"\b{re.escape(replace_obj)}\b", neg_obj, gt_caption, count=1)
            
            # make sure it is not the same as some other negative or the ground truth caption before adding to the negative list
            if negative in negatives or negative == gt_caption:
                continue
            negatives.append(negative)
            
            # Reweight the object weights by reducing the weight of the used object
            ind = objs.index(replace_obj)
            weights[ind] /= 2
        
        neg_captions[complexity] = negatives
    
    # dump the map of complexity to negative captions  
    with open(f"{outfolder}/{file}", "w") as f:
        json.dump(neg_captions, f, indent=4)
        
def get_all_rels(file):
    ''' Get all the relations being used in the file, includes all relations from the scene 
        file: Name of the file
    '''
    folder = "subgraphs_dict"
    with open(f"{folder}/{file}", "r") as f:
        data = json.load(f)
    rel_list = [x['predicate'] for x in data['graph']['relationships']]
    relations = []
    for rel in rel_list:
        relations.extend(rel)
    return relations
        
def get_rel_captions(file, outfolder, rel_map = {}, infolder="subgraphs_caption", gt_caption_folder="gt_captions"):   
    
    # if the output file already exists do not run the code again
    if os.path.exists(f"{outfolder}/{file}"):
        return
       
    # load the data from the subgraph captions dict
    with open(f"{infolder}/{file}", "r") as f:
        data = json.load(f)
       
    # check if we already have ground truth caption, if not query gpt
    if not os.path.exists(f"{gt_caption_folder}/{file}"):
        gt_captions(data, f"{gt_caption_folder}/{file}")
    
    # load the ground truth captions from the file
    with open(f"{gt_caption_folder}/{file}", "r") as f:
        gt_captions_data = json.load(f)
        
    # if no ground truth captions to work with just return
    if gt_captions_data == {}:
        return

    neg_captions = {} # initialize negative caption list
    
    # get all the relations being used in order to avoid them in hard negative captions 
    all_relations = get_all_rels(file)
        
    # get the negatives
    for complexity, cap_dicts in data.items():
        
        # get the possible objects 
        rels = []
        for cap_dict in cap_dicts:
            if cap_dict["rel"] != "":
                rels.append(cap_dict["rel"].strip())
                
        # create a weights list for all the objects 
        weights = [1 for _ in range(len(rels))]
                
        # if there is no ground truth caption or there are no relations in the caption just continue
        if complexity not in gt_captions_data or rels == []:
            continue
        
        gt_caption = gt_captions_data[complexity]
        negatives = []
        i = 0
        while len(negatives) < 4:
            i += 1
            
            # do max 10 iterations 
            if i > 10:
                break
            
            # choose a random relation from the list 
            replace_rel = random.choices(rels, weights)[0]
            
            # check if we already have a list of negative relations to replace this one
            if replace_rel in rel_map:
                neg_objs = rel_map[replace_rel]
            else:
                try:
                    neg_objs = ast.literal_eval(get_neg_rel(replace_rel))
                except:
                    continue
                rel_map[replace_rel] = neg_objs
                
            # don't accept any relation present in any of the captions for this image
            neg_rels_poss = [item for item in neg_objs if item not in all_relations]
            
            # choose at random an attribute and replace it in ground truth caption to get a negative
            if neg_rels_poss == []:
                continue
            neg_rel = random.choice(neg_rels_poss)
            negative = re.sub(rf"\b{re.escape(replace_rel)}\b", neg_rel, gt_caption, count=1)
            
            # make sure it is not the same as some other negative or the ground truth caption before adding to the negative list
            if negative in negatives or negative == gt_caption:
                continue
            negatives.append(negative)
            
            # Reweight the relation weights by reducing the weight of the used relation
            ind = rels.index(replace_rel)
            weights[ind] /= 2
        
        neg_captions[complexity] = negatives
    
    # dump the map of complexity to negative captions  
    with open(f"{outfolder}/{file}", "w") as f:
        json.dump(neg_captions, f, indent=4)
        
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument('-t', '--type', type=str, help="type of negative", default="attribute")
    parser.add_argument('-i', '--infolder', type=str, help="input folder", default="subgraphs_caption")
    parser.add_argument('-o', '--outfolder', type=str, help="output folder", required=True)
    parser.add_argument('-c', '--gt-caption', type=str, help="ground truth caption folder", default="gt_captions")
    parser.add_argument('-att', '--attribute', type=str, help="map of attributes to negatives", default="attribute_map.json")
    parser.add_argument('-obj', '--object', type=str, help="map of objects to negatives", default="object_map.json")
    parser.add_argument('-rel', '--relation', type=str, help="map of relation to negatives", default="relation_map.json")
    parser.add_argument('-s', '--start', type=int, help="Starting file number", default=0)
    parser.add_argument('-e', '--end', type=int, help="Ending file number", default=10000)

    # Parse arguments
    args = parser.parse_args()
    
    # get all the attributes
    infolder = args.infolder
    outfolder = args.outfolder
    captions = args.gt_caption
    

    # make the outfolder if it does not exist
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
        
    # make the caption folder if it does not exist
    if not os.path.exists(captions):
        os.makedirs(captions)
        
    # if the type was attribute
    if args.type == "attribute":
        attr_map_path = args.attribute
        # create an attribute json if it does not exist already
        if os.path.exists(attr_map_path):
            with open(attr_map_path, "r") as f:
                attr_map = json.load(f)
        else:
            attr_map = {}

        files = os.listdir(infolder)
        for file in tqdm(files[args.start:args.end]):
            get_attr_captions(file, outfolder, infolder=infolder, attr_map=attr_map)
            
        with open(attr_map_path, "w") as f:
            json.dump(attr_map, f, indent=4) 
            
    # if the type was object
    elif args.type == "object":
        obj_map_path = args.object
        # create an object json if it does not exist already
        if os.path.exists(obj_map_path):
            with open(obj_map_path, "r") as f:
                obj_map = json.load(f)
        else:
            obj_map = {}

        files = os.listdir(infolder)
        for file in tqdm(files[args.start:args.end]):
            get_obj_captions(file, outfolder, infolder=infolder, obj_map=obj_map)
            
        with open(obj_map_path, "w") as f:
            json.dump(obj_map, f, indent=4) 
            
    # if the type was relation
    elif args.type == "relation":
        rel_map_path = args.relation
        # create an object json if it does not exist already
        if os.path.exists(rel_map_path):
            with open(rel_map_path, "r") as f:
                rel_map = json.load(f)
        else:
            rel_map = {}

        files = os.listdir(infolder)
        for file in tqdm(files[args.start:args.end]):
            get_rel_captions(file, outfolder, infolder=infolder, rel_map=rel_map)
            
        with open(rel_map_path, "w") as f:
            json.dump(rel_map, f, indent=4) 
        
if __name__ == "__main__":
    main()