import open_clip
import torch
from PIL import Image
import argparse
import os, json
import random
from tqdm import tqdm
import numpy as np

model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')

def clip_scores(url, text_array):
    image = Image.open(url)
    image = preprocess(image).unsqueeze(0)
    text = tokenizer(text_array)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return text_probs

def get_results(gt_captions, negative_captions, outfolder):

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    files = os.listdir(negative_captions)

    images1 = os.listdir("visual_genome_python_driver/VG_100K")

    for file in tqdm(files):
        if os.path.exists(f"{outfolder}/{file}"):
            continue
        image_name = f"{file.split('.')[0]}.jpg"
        if image_name in images1:
            img_path = f"visual_genome_python_driver/VG_100K/{image_name}"
        else:
            img_path = f"visual_genome_python_driver/VG_100K_2/{image_name}"
        
        with open(f"{negative_captions}/{file}", "r") as f:
            negatives = json.load(f)
            
        with open(f"{gt_captions}/{file}", "r") as f:
            gt = json.load(f)
        
        results = {}
        
        for complexity, negs in negatives.items():
            caption = gt[complexity]
            if len(negs) == 4:
                ind = random.randint(1,len(negs))
                negs.insert(ind, caption)
                text_probs = clip_scores(img_path, negs)
                results[complexity] = (negs, text_probs.tolist(), ind)
            
        with open(f"{outfolder}/{file}", "w") as f:
            json.dump(results, f, indent=4)


def get_accuracy(folder, k=1):
    files = os.listdir(folder)
    
    correct = {}
    total = {}
    
    complexities = [f'{x}' for x in range(3, 13)]
    for c in complexities:
        correct[c] = 0
        total[c] = 0
    
    for file in tqdm(files):
        with open(f"{folder}/{file}", "r") as f:
            results = json.load(f)
            
        for c, info in results.items():
            _, text_probs, ind = info
            sorted_ids = np.argsort(text_probs[0])[::-1]
            check_array = sorted_ids[:k]
            if ind in check_array:
                correct[c] += 1
            total[c] += 1
     
    accuracy = {}       
    for c in complexities:
        if total[c] != 0:
            accuracy[c] = correct[c]/total[c]
        else:
            accuracy[c] = 0
        
    print("Accuracy: ", accuracy)
    print("Total: ", total)
 
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument('-gt', '--gt-captions', type=str, help="folder to get the groundtruths", default="gt_captions")
    parser.add_argument('-n', '--negatives', type=str, help="folder to get the hard negatives", default="obj_negs")
    parser.add_argument('-o', '--outfolder', type=str, help="folder to store the results from clip", default="clip_results_obj")
    parser.add_argument('-k', '--recall-k', type=int, help="Recall at k that should be calculated, max 5", default=1)
    
    # Parse arguments
    args = parser.parse_args()
    
    # get_results(gt_captions=args.gt_captions, negative_captions=args.negatives, outfolder=args.outfolder)
    get_accuracy(k=1, folder=args.outfolder)
    get_accuracy(k=2, folder=args.outfolder)
    get_accuracy(k=3, folder=args.outfolder)


if __name__ == "__main__":
    main()


            
        
            
        