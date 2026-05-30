"""
Composed caption retrieval evaluation on COMPASS compositional negatives.

Ground-truth captions and composed negative captions are loaded from
complexity-keyed JSON files. For each image, the GT caption is inserted
at a random position among the negatives, and the selected vision-language
model computes image-text similarity scores for all candidates.

Results store the candidate captions, similarity scores, and GT index.
Accuracy can be measured as Recall@K by checking whether the GT caption
appears among the top-K ranked captions for each complexity level.
"""

import torch
from PIL import Image
import argparse
import os, json
import random
from tqdm import tqdm
import numpy as np
import open_clip
from transformers import AutoProcessor, AutoModel, BlipProcessor, BlipForImageTextRetrieval
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# Model will be initialized based on command-line argument
model = None
preprocess = None
tokenizer = None
processor = None
device = None
model_type = None
use_transformers = False

IMAGE_PATH = "VISUAL_GENOME/images"

def initialize_model(model_name='clip'):
    """Initialize the model based on the model name using OpenCLIP or Transformers."""
    global model, preprocess, tokenizer, processor, device, model_type, use_transformers
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = model_name
    use_transformers = False
    
    if model_name == 'clip':
        model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
        tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
    elif model_name == 'siglipv2':
        use_transformers = True
        model_id = "google/siglip2-so400m-patch14-384"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
    elif model_name == 'peclip':
        model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
        preprocess = transforms.get_image_transform(model.image_size)
        tokenizer = transforms.get_text_tokenizer(model.context_length)
        print("Initialized PE-CLIP model successfully.")
    elif model_name == 'blip':
        use_transformers = True
        model_id = "Salesforce/blip-itm-large-coco"
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForImageTextRetrieval.from_pretrained(model_id)
        print("Initialized BLIP model successfully.")
    elif model_name == "negclip":
        path = os.path.join("models", "negclip.pth")
        if not os.path.exists(path):
            print("Downloading the NegCLIP model...")
            import gdown
            gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=path, device=device, weights_only=False)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model = model.eval().to(device)
        print("Initialized NegCLIP model successfully.")
    elif model_name == 'ceclip':
        path = os.path.join("models", "CE_CLIP.pt")
        if not os.path.exists(path):
            print("Downloading CE-CLIP model from HuggingFace...")
            from huggingface_hub import hf_hub_download
            os.makedirs("models", exist_ok=True)
            hf_hub_download(
                repo_id="le723z/CE_CLIP",
                filename="CE_CLIP.pt",
                local_dir="models"
            )
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained=path, device=device, weights_only=False
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model = model.eval().to(device)
        print("Initialized CE-CLIP (ViT-B-32) model successfully.")
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'clip', 'siglipv2', 'peclip', or 'blip'.")
    
    model = model.to(device)
    model.eval()
    print(f"Initialized {model_name} model successfully on {device}.")
    
def negclip_scores(url, text_array):
    """Compute similarity scores using NegCLIP model."""
    image = Image.open(url).convert('RGB')
    
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        text_tokens = tokenizer(text_array).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T)
        
        text_probs = similarity.softmax(dim=-1)
    
    return text_probs.cpu()

def siglip_scores(url, text_array):
    """Compute similarity scores using SigLIP v2 with Transformers library."""
    image = Image.open(url).convert('RGB')
    
    with torch.no_grad():
        # inputs = processor(text=text_array, images=image, return_tensors="pt", 
        #                  padding="max_length", max_length=64).to(device)

        inputs = processor(text=text_array, images=image, return_tensors="pt", 
                          padding=True, truncation=True).to(device)
        
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        text_probs = torch.sigmoid(logits_per_image)
    
    return text_probs.cpu()

def clip_scores(url, text_array):
    """Compute similarity scores using CLIP or SigLIP v2 with OpenCLIP library."""
    image = Image.open(url).convert('RGB')
    
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        text_tokens = tokenizer(text_array).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T)
        
        text_probs = similarity.softmax(dim=-1)
    
    return text_probs.cpu()

def peclip_scores(url, text_array):
    """Compute similarity scores using PE-CLIP model."""
    image = Image.open(url).convert('RGB')
    
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_tokens = tokenizer(text_array).to(device)
    
    with torch.no_grad(), torch.autocast("cuda"):
        image_features, text_features, logit_scale = model(image_input, text_tokens)
        text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
    
    return text_probs.cpu()

def blip_scores(url, text_array):
    """Compute similarity scores using BLIP model."""
    image = Image.open(url).convert('RGB')
    
    with torch.no_grad():
        # Process each text with the image
        scores_list = []
        for text in text_array:
            inputs = processor(image, text, return_tensors="pt").to(device)
            outputs = model(**inputs, use_itm_head=True)
            # ITM score (image-text matching) - higher means better match
            itm_score = outputs.itm_score[:, 1]  # Index 1 is the "match" score
            scores_list.append(itm_score.item())
        
        # Convert to tensor and apply softmax
        scores_tensor = torch.tensor(scores_list).unsqueeze(0)
        text_probs = torch.softmax(scores_tensor, dim=-1)
    
    return text_probs.cpu()

def scores(url, text_array):
    """
    Compute similarity scores between an image and its corresponding text descriptions.
    """
    
    if model_type == 'siglipv2':
        return siglip_scores(url, text_array)
    elif model_type == 'clip':
        return clip_scores(url, text_array)
    elif model_type == 'peclip':
        return peclip_scores(url, text_array)
    elif model_type == 'blip':
        return blip_scores(url, text_array)
    elif model_type == 'negclip':
        return negclip_scores(url, text_array)
    elif model_type == 'ceclip':
        return clip_scores(url, text_array)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_results(gt_captions, negative_captions, outfolder, s=0, e=100):

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
        
    files = os.listdir(negative_captions)

    count = {c: 0 for c in range(2, 13)}

    for file in tqdm(files[s:e]):

        out_path = f"{outfolder}/{file}"
        
        if os.path.exists(out_path):
            continue
        else:
            results = {}

        image_name = f"{file.split('.')[0]}.jpg"
        img_path = f"{IMAGE_PATH}/{image_name}"
        
        try:
            with open(f"{negative_captions}/{file}", "r") as f:
                negatives = json.load(f)
        except:
            print(f"Error reading negative captions file: {file}")
            continue
        
        try:   
            with open(f"{gt_captions}/{file}", "r") as f:
                gt = json.load(f)
        except:
            print(f"Error reading ground truth file: {file}")
            continue
        
        for complexity, negs in negatives.items():

            caption = gt[complexity]

            if len(negs) == int(complexity):
                negs_copy = negs.copy()  # avoid modifying original
                ind = random.randint(1, len(negs_copy))
                negs_copy.insert(ind, caption)

                text_probs = scores(img_path, negs_copy)

                results[complexity] = (negs_copy, text_probs.tolist(), ind)
                count[int(complexity)] += 1

        with open(out_path, "w") as f:
                json.dump(results, f, indent=4)


def get_accuracy(folder, k=1, s=0, e=100, cl=1, ch=12):
    files = os.listdir(folder)
    
    correct = {}
    total = {}
    
    complexities = [f'{x}' for x in range(cl, ch+1)]
    for c in complexities:
        correct[c] = 0
        total[c] = 0
    
    for file in tqdm(files[s:e]):
        try:
            with open(f"{folder}/{file}", "r") as f:
                results = json.load(f)
        except:
            os.remove(f"{folder}/{file}")
            continue
            
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

    parser.add_argument(
        '-m', '--model',
        type=str,
        choices=['clip', 'siglipv2', 'peclip', 'blip', 'negclip', 'ceclip'],
        help="Model to use for evaluation",
        default='clip'
    )

    parser.add_argument(
        '-sl', '--structural-level',
        type=str,
        choices=['l3', 'l2_or', 'l2_oa'],
        required=True,
        help="Structural level to evaluate"
    )

    parser.add_argument(
        '-o', '--outfolder',
        type=str,
        help="folder to store the results from clip",
        default="clip_results_obj"
    )

    parser.add_argument(
        '-k', '--recall-k',
        type=int,
        help="Recall at k that should be calculated, max 5",
        default=1
    )

    parser.add_argument(
        '-s', '--start',
        type=int,
        help="Starting file number",
        default=0
    )

    parser.add_argument(
        '-e', '--end',
        type=int,
        help="Ending file number",
        default=10000
    )

    parser.add_argument(
        '-cl', '--complexity_low',
        type=int,
        help="Lowest Complexity",
        default=1
    )

    parser.add_argument(
        '-ch', '--complexity_high',
        type=int,
        help="Highest Complexity",
        default=12
    )

    # Parse arguments
    args = parser.parse_args()

    # Map structural level to gt captions + negatives
    structural_mapping = {
        'l3': {
            'gt_captions': 'COMPASS/gt-caption/l3',
            'negatives': 'COMPASS/compositional-integration/l3/composed'
        },
        'l2_or': {
            'gt_captions': 'COMPASS/gt-caption/l2-OR',
            'negatives': 'COMPASS/compositional-integration/l2-OR/composed'
        },
        'l2_oa': {
            'gt_captions': 'COMPASS/gt-caption/l2-OA',
            'negatives': 'COMPASS/compositional-integration/l2-OA/composed'
        }
    }

    gt_captions = structural_mapping[args.structural_level]['gt_captions']
    negatives = structural_mapping[args.structural_level]['negatives']

    print("Running composed evaluation on decomposed captions with model:", args.model)

    # Initialize the selected model
    initialize_model(args.model)

    get_results(
        gt_captions=gt_captions,
        negative_captions=negatives,
        outfolder=args.outfolder,
        s=args.start,
        e=args.end
    )

    # get_accuracy(
    #     k=args.recall_k,
    #     folder=args.outfolder,
    #     s=args.start,
    #     e=args.end,
    #     cl=args.complexity_low,
    #     ch=args.complexity_high
    # )
    

if __name__ == "__main__":
    main()


            
        
            
        