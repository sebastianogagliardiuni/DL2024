import argparse
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
import wandb
import json

from utils import load_data, ROOT_IMAGENET_A, ROOT_IMAGENET_V2, get_device, imagenet_a_mask, imagenet_v2_mask
from models import load_ResNet50, load_VITB16
from tta import MEMOAdaptiveBatchNormModel

# Baseline -- Test model performances with the default protocol (one forward pass without TTA)
def test_baseline(model, test_loader, label_mask, device):

    # Results
    hyp = []
    ref = []
    
    # Evaluation mode
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Iterate over batches 
        for batch_idx, (input, target) in enumerate(tqdm(test_loader)):
            
            # Load data into GPU
            input = input.to(device)
    
            # Model inference
            output = model(input)
            # Align ImageNet prediction with ImageNet-A/V2
            output = output[:, label_mask]
    
            # Save results
            hyp.extend([label for label in torch.argmax(output, dim=1).tolist()])
            ref.extend([gt for gt in target.tolist()])
    
    return hyp, ref


# TESTS

# ResNet50 tests
def test_resnet50(norm_adaptation, prior_N, device, wb):

    #  - ResNet-50 -
    print("-- ResNet-50 --\n")

    # Load model
    print("Loading ResNet-50 ...")
    model_resnet50, preprocess_resnet50 = load_ResNet50()
    print()

    # Norm adaptation
    if norm_adaptation:
        print(f"Adapting Normalization Layers, with adaptation factor {prior_N}")
        model_resnet50 = MEMOAdaptiveBatchNormModel(model_resnet50, prior_N=prior_N, keep_test_running=False)
    
    # ImageNet-A
    print("ImageNet-A")
    # Load dataset
    imagenet_a_resnet50 = load_data(ROOT_IMAGENET_A, transform=preprocess_resnet50)
    imagenet_a_loader_resnet50 = torch.utils.data.DataLoader(imagenet_a_resnet50, BATCH_SIZE)
    # Test model
    baseline_hyp_resnet50_imagenet_a, baseline_ref_resnet50_imagenet_a = test_baseline(model_resnet50, imagenet_a_loader_resnet50,
                                                                                       imagenet_a_mask, DEVICE)
    baseline_resnet50_imagenet_a_report = classification_report(baseline_ref_resnet50_imagenet_a, 
                                                                baseline_hyp_resnet50_imagenet_a, 
                                                                zero_division=False, output_dict=True)
    # Results
    print(f"ImageNet-A :- Accuracy: {baseline_resnet50_imagenet_a_report['accuracy'] * 100}")
    if wb:
        wandb.log({"ImageNet-A-acc": baseline_resnet50_imagenet_a_report['accuracy'] * 100})

    # Clean memory
    del imagenet_a_resnet50
    del imagenet_a_loader_resnet50

    # ImageNet-V2
    print("ImageNet-V2")
    # Load dataset
    imagenet_v2_resnet50 = load_data(ROOT_IMAGENET_V2, transform=preprocess_resnet50)
    imagenet_v2_loader_resnet50 = torch.utils.data.DataLoader(imagenet_v2_resnet50, BATCH_SIZE)
    # Test model
    baseline_hyp_resnet50_imagenet_v2, baseline_ref_resnet50_imagenet_v2 = test_baseline(model_resnet50, imagenet_v2_loader_resnet50,
                                                                                         imagenet_v2_mask, DEVICE)
    baseline_resnet50_imagenet_v2_report = classification_report(baseline_ref_resnet50_imagenet_v2, 
                                                                 baseline_hyp_resnet50_imagenet_v2, 
                                                                 zero_division=False, output_dict=True)

    # Results
    print(f"ImageNet-V2 :- Accuracy: {baseline_resnet50_imagenet_v2_report['accuracy'] * 100}")
    if wb:
        wandb.log({"ImageNet-V2-acc": baseline_resnet50_imagenet_v2_report['accuracy'] * 100})

# VIT-B/16 tests
def test_vit_b_16(device, wb):

    # VIT-B/16
    
    print("\n-- VIT-B/16 --\n")

    # Load model
    print("Loading VIT-B/16 ...")
    model_vit_b_16, preprocess_vit_b_16 = load_VITB16()
    print()
    
    # ImageNet-A
    print("ImageNet-A")
    # Load dataset
    imagenet_a_vit_b_16 = load_data(ROOT_IMAGENET_A, transform=preprocess_vit_b_16)
    imagenet_a_loader_vit_b_16 = torch.utils.data.DataLoader(imagenet_a_vit_b_16, BATCH_SIZE)
    # Test model
    baseline_hyp_vit_b_16_imagenet_a, baseline_ref_vit_b_16_imagenet_a = test_baseline(model_vit_b_16, imagenet_a_loader_vit_b_16, 
                                                                                       imagenet_a_mask, DEVICE)
    baseline_vit_b_16_imagenet_a_report = classification_report(baseline_ref_vit_b_16_imagenet_a, 
                                                                baseline_hyp_vit_b_16_imagenet_a, 
                                                                zero_division=False, output_dict=True)
    print(f"ImageNet-A :- Accuracy: {baseline_vit_b_16_imagenet_a_report['accuracy'] * 100}")
    if wb:
        wandb.log({"ImageNet-A-acc": baseline_vit_b_16_imagenet_a_report['accuracy'] * 100})

    # Clean memory
    del imagenet_a_vit_b_16
    del imagenet_a_loader_vit_b_16
    
    # ImageNet-V2
    print("ImageNet-V2")
    # Load dataset
    imagenet_v2_vit_b_16 = load_data(ROOT_IMAGENET_V2, transform=preprocess_vit_b_16)
    imagenet_v2_loader_vit_b_16 = torch.utils.data.DataLoader(imagenet_v2_vit_b_16, BATCH_SIZE)
    # Test model
    baseline_hyp_vit_b_16_imagenet_v2, baseline_ref_vit_b_16_imagenet_v2 = test_baseline(model_vit_b_16, imagenet_v2_loader_vit_b_16, 
                                                                                         imagenet_v2_mask, DEVICE)
    baseline_vit_b_16_imagenet_v2_report = classification_report(baseline_ref_vit_b_16_imagenet_v2, 
                                                                 baseline_hyp_vit_b_16_imagenet_v2, 
                                                                 zero_division=False, output_dict=True)

    # Results
    print(f"ImageNet-V2 :- Accuracy: {baseline_vit_b_16_imagenet_v2_report['accuracy'] * 100}")
    if wb:
        wandb.log({"ImageNet-V2-acc": baseline_vit_b_16_imagenet_v2_report['accuracy'] * 100})

# MAIN
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Architecture
    parser.add_argument('-m', '--model', type=str, choices=['ResNet-50', 'ViT-B/16'], help="Backbone")
    parser.add_argument('--adapt', action='store_true', default=False, help="Adapt model (Norm adaptation)")
    # Hyperparams
    parser.add_argument('--prior_N', type=int, default=16, help="Norm adaptation factor")
    # Logs
    parser.add_argument('--wandb', action='store_true', default=False, help="Log on W&B")
    args = parser.parse_args()
    
    print("--- BASELINE ---")
    
    # GPU
    DEVICE = get_device()
    print(f"Using device: {DEVICE}\n")
    
    BATCH_SIZE = 1

    assert (not args.adapt or BATCH_SIZE == 1)

    # Norm adaptation
    norm_adaptation = args.adapt
    norm_adaptation_tag = "+NA" if norm_adaptation else ""

    # Hyperparams
    prior_N = args.prior_N
    
    # W&B setup
    wb = args.wandb
    if wb:
        wandb.require("core")
        wandb.login()
        with open('../wandb-config.json', 'r') as file:
            config = json.load(file)
        project_name = config.get('project-name')

    # Run tests
    model_name = args.model
    if model_name == "ResNet-50":
        if wb:
            wandb.init(project=project_name, name=("Baseline" + norm_adaptation_tag + "_ResNet50"), config={})
        test_resnet50(norm_adaptation, prior_N, DEVICE, wb)
    elif model_name == "ViT-B/16":
        if wb:
            wandb.init(project=project_name, name=("Baseline_ViT-B/16"), config={})
        test_vit_b_16(DEVICE, wb)

