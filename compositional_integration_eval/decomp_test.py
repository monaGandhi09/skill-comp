"""
Decomposed caption retrieval evaluation on COMPASS compositional negatives.

Each JSON file contains complexity-keyed lists of caption sets of the form:
    {
        "GT caption": ["neg1", "neg2", ...],
        "replaced": "object|relation|attribute"
    }

For each image, the GT caption and its corresponding negative captions are
scored using the selected vision-language model. The GT caption is always
placed at index 0, and evaluation checks whether it receives the highest
image-text similarity score among all candidates.

Results store the captions, similarity scores, correctness indicator, and
the compositional component (object, relation, or attribute) that was
replaced to generate the negatives.
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

    print(f"Initializing {model_name} model on {device}...")

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
        inputs = processor(text=text_array, images=image, return_tensors="pt", 
                         padding="max_length", max_length=64).to(device)
        
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
    # print(f"Computing BLIP scores for {url} with texts: {text_array}")
    
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

def get_results_decomposed(infolder, outfolder, s=0, e=10, noise=False):
        
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
        
    files = os.listdir(infolder)

    for file in tqdm(files[s:e]):

        out_path = f"{outfolder}/{file}"

        if os.path.exists(out_path):
            continue
        else:
            result_dict = {}

        image_name = f"{file.split('.')[0]}.jpg"
        img_path = f"{IMAGE_PATH}/{image_name}"
        
        try:
            with open(f"{infolder}/{file}", "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading input {file}: {e}")
            continue

        for complexity, dicts in data.items():

            results = []

            for dictionary in dicts:
                gt_caption = next(k for k in dictionary.keys() if k.startswith("There is"))
                negative_captions = dictionary[gt_caption]

                text_array = [gt_caption] + negative_captions
                score_vals = scores(img_path, text_array).cpu().numpy()[0]
                
                result = {
                    "gt_caption": gt_caption,
                    "negative_captions": negative_captions,
                    "scores": score_vals.tolist(),
                    "correct": True if np.argmax(score_vals) == 0 else False,
                    "replaced": dictionary["replaced"]
                }

                results.append(result)

            if results:
                result_dict[complexity] = results

        with open(out_path, "w") as f:
                json.dump(result_dict, f, indent=4)
            


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
        required=True,
        help="Folder to save output results"
    )

    parser.add_argument(
        '-k', '--recall-k',
        type=int,
        default=1,
        help="Recall@K value"
    )

    parser.add_argument(
        '-s', '--start',
        type=int,
        default=0,
        help="Start index for processing files"
    )

    parser.add_argument(
        '-e', '--end',
        type=int,
        default=1,
        help="End index for processing files"
    )

    parser.add_argument(
        '-cl', '--complexity_low',
        type=int,
        default=1,
        help="Lowest complexity level"
    )

    parser.add_argument(
        '-ch', '--complexity_high',
        type=int,
        default=12,
        help="Highest complexity level"
    )

    # Parse arguments
    args = parser.parse_args()

    # Map structural level to negatives folder
    structural_mapping = {
        'l3': {
            'negatives': 'COMPASS/compositional-integration/l3/decomposed',
        },
        'l2_or': {
            'negatives': 'COMPASS/compositional-integration/l2-OR/decomposed',
        },
        'l2_oa': {
            'negatives': 'COMPASS/compositional-integration/l2-OA/decomposed',
        }
    }

    negatives = structural_mapping[args.structural_level]['negatives']

    print("Running decomposed evaluation with model:", args.model)

    # Initialize the selected model
    initialize_model(args.model)

    get_results_decomposed(
        infolder=negatives,
        outfolder=args.outfolder,
        s=args.start,
        e=args.end
    )

    # # Optional Recall@K evaluation
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