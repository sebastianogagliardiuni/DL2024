import argparse
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
import wandb
import json
import os
import copy

from utils import load_data, ROOT_IMAGENET_A, ROOT_IMAGENET_V2, get_device, imagenet_a_mask, imagenet_v2_mask, collate_fn_augs, get_augmented_preprocessing
from models import load_ResNet50, load_VITB16
from tta import MEMOAdaptiveBatchNormModel, RMEMO_inference, get_augmentation, get_optimizer

# TOPK MEMO -- Test model performances using TTA TOPK MEMO inference
def test_memo(model, test_loader, label_mask, hyperparam, device):

    # Results
    hyp = []
    ref = []

    # Iterate over samples
    for batch_idx, batch in enumerate(tqdm(test_loader)):

        # Decompose batch and load into GPU
        input, aug_imgs, target = batch["x"], batch["aug_x"], batch["target"]
        input, aug_imgs = input.to(device), aug_imgs.to(device)

        # Fork a new model with the same pre-trained parameters
        tmp_model = copy.deepcopy(model)

        # Define fresh optimizer
        optim_hyperparam = {"lr": hyperparam['lr']}
        optimizer = get_optimizer(tmp_model, hyperparam['optim'], optim_hyperparam)
        
        # Model inference
        output = RMEMO_inference(tmp_model, input, aug_imgs, hyperparam['conf_tresh'], hyperparam['heuristic'], 
                                 hyperparam['sel_K'], hyperparam['bias_c'],  optimizer, device)
        # Align ImageNet prediction with ImageNet-A/V2
        output = output[:, label_mask]

        # Save results
        hyp.extend([label for label in torch.argmax(output, dim=1).tolist()])
        ref.extend([target.item()])

    return hyp, ref

# TESTS

# ResNet50 tests
def test_resnet50(hyperparam, A, norm_adaptation, device, wb):

    #  - ResNet-50 -
    print("-- ResNet-50 --\n")

    # Load model
    print("Loading ResNet-50 ...")
    model_resnet50, preprocess_resnet50 = load_ResNet50()
    print()

    # Prepare augmented preprocessing
    A = get_augmented_preprocessing(A, preprocess_resnet50)

    # Norm adaptation
    if norm_adaptation:
        prior_N = hyperparam['prior_N']
        print(f"Adapting Normalization Layers, with adaptation factor {prior_N}")
        if hyperparam['use_batch_stats']:
            print(f"Using collected test batch statistics in 1st forward pass for 2nd forward pass")
        model_resnet50 = MEMOAdaptiveBatchNormModel(model_resnet50, prior_N=prior_N, 
                                                    keep_test_running=hyperparam['use_batch_stats'])
    
    # ImageNet-A
    print("ImageNet-A")
    # Load dataset
    imagenet_a_resnet50 = load_data(ROOT_IMAGENET_A, transform=None)
    imagenet_a_loader_resnet50 = torch.utils.data.DataLoader(imagenet_a_resnet50, batch_size=1, num_workers=os.cpu_count(),
                                                             collate_fn= lambda data: collate_fn_augs(
                                                                 data, A, hyperparam["N"], preprocess_resnet50)
                                                             )
    # Test model
    memo_hyp_resnet50_imagenet_a, memo_ref_resnet50_imagenet_a = test_memo(model_resnet50,
                                                                           imagenet_a_loader_resnet50,
                                                                           imagenet_a_mask, 
                                                                           hyperparam, DEVICE)
    memo_resnet50_imagenet_a_report = classification_report(memo_ref_resnet50_imagenet_a, memo_hyp_resnet50_imagenet_a, 
                                                            zero_division=False, output_dict=True)

    # Results
    print(f"ImageNet-A :- Accuracy: {memo_resnet50_imagenet_a_report['accuracy'] * 100}")
    if wb:
        wandb.log({"ImageNet-A-acc": memo_resnet50_imagenet_a_report['accuracy'] * 100})

    # Clean memory
    del imagenet_a_resnet50
    del imagenet_a_loader_resnet50
    
    # ImageNet-V2
    print("ImageNet-V2")
    # Load dataset
    imagenet_v2_resnet50 = load_data(ROOT_IMAGENET_V2, transform=None)
    imagenet_v2_loader_resnet50 = torch.utils.data.DataLoader(imagenet_v2_resnet50, batch_size=1, num_workers=os.cpu_count(),
                                                              collate_fn= lambda data: collate_fn_augs(
                                                                 data, A, hyperparam["N"], preprocess_resnet50)
                                                              )
    # Test model 
    memo_hyp_resnet50_imagenet_v2, memo_ref_resnet50_imagenet_v2 = test_memo(model_resnet50,
                                                                             imagenet_v2_loader_resnet50,
                                                                             imagenet_v2_mask, 
                                                                             hyperparam, DEVICE)
    memo_resnet50_imagenet_v2_report = classification_report(memo_ref_resnet50_imagenet_v2,  memo_hyp_resnet50_imagenet_v2, 
                                                             zero_division=False, output_dict=True)

    # Results
    print(f"ImageNet-V2 :- Accuracy: {memo_resnet50_imagenet_v2_report['accuracy'] * 100}")
    if wb:
        wandb.log({"ImageNet-V2-acc": memo_resnet50_imagenet_v2_report['accuracy'] * 100})

# VIT-B/16 tests
def test_vit_b_16(hyperparam, A, device, wb):

    # VIT-B/16
    
    print("\n-- VIT-B/16 --\n")

    # Load model
    print("Loading VIT-B/16 ...")
    model_vit_b_16, preprocess_vit_b_16 = load_VITB16()
    print()

    # Prepare augmented preprocessing
    A = get_augmented_preprocessing(A, preprocess_vit_b_16)
    
    # ImageNet-A
    print("ImageNet-A")
    # Load dataset
    imagenet_a_vit_b_16 = load_data(ROOT_IMAGENET_A, transform=None)
    imagenet_a_loader_vit_b_16 = torch.utils.data.DataLoader(imagenet_a_vit_b_16, batch_size=1, num_workers=os.cpu_count(),
                                                             collate_fn= lambda data: collate_fn_augs(
                                                                 data, A, hyperparam["N"], preprocess_vit_b_16)
                                                             )
    # Test model
    memo_hyp_vit_b_16_imagenet_a, memo_ref_vit_b_16_imagenet_a = test_memo(model_vit_b_16,
                                                                           imagenet_a_loader_vit_b_16,
                                                                           imagenet_a_mask, 
                                                                           hyperparam, DEVICE)
    memo_vit_b_16_imagenet_a_report = classification_report(memo_ref_vit_b_16_imagenet_a, memo_hyp_vit_b_16_imagenet_a, 
                                                            zero_division=False, output_dict=True)

    # Results
    print(f"ImageNet-A :- Accuracy: {memo_vit_b_16_imagenet_a_report['accuracy'] * 100}")
    if wb:
        wandb.log({"ImageNet-A-acc": memo_vit_b_16_imagenet_a_report['accuracy'] * 100})

    # Clean memory
    del imagenet_a_vit_b_16
    del imagenet_a_loader_vit_b_16

    # ImageNet-V2
    print("ImageNet-V2")
    # Load dataset
    imagenet_v2_vit_b_16 = load_data(ROOT_IMAGENET_V2, transform=None)
    imagenet_v2_loader_vit_b_16 = torch.utils.data.DataLoader(imagenet_v2_vit_b_16, batch_size=1, num_workers=os.cpu_count(),
                                                              collate_fn= lambda data: collate_fn_augs(
                                                                 data, A, hyperparam["N"], preprocess_vit_b_16)
                                                              )
    # Test model
    memo_hyp_vit_b_16_imagenet_v2, memo_ref_vit_b_16_imagenet_v2 = test_memo(model_vit_b_16,
                                                                             imagenet_v2_loader_vit_b_16,
                                                                             imagenet_v2_mask, 
                                                                             hyperparam, DEVICE)
    memo_vit_b_16_imagenet_v2_report = classification_report(memo_ref_vit_b_16_imagenet_v2,  memo_hyp_vit_b_16_imagenet_v2, 
                                                             zero_division=False, output_dict=True)

    # Results
    print(f"ImageNet-V2 :- Accuracy: {memo_vit_b_16_imagenet_v2_report['accuracy'] * 100}")
    if wb:
        wandb.log({"ImageNet-V2-acc": memo_vit_b_16_imagenet_v2_report['accuracy'] * 100})


# MAIN
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Architecture
    parser.add_argument('-m', '--model', type=str, choices=['ResNet-50', 'ViT-B/16'], help="Backbone")
    parser.add_argument('--adapt', action='store_true', default=False, help="Adapt model (Norm adaptation)")
    parser.add_argument('--use_batch_stats', action='store_true', default=False, 
                        help="Use batch statistics of 1st forward pass in 2nd forward pass")
    # Hyperparams
    parser.add_argument('-a', '--aug_type', type=str, choices=['AugMix', 'RandomResizedCrop', 'Mixture'], help="Augmentation strategy")
    parser.add_argument('-N', '--aug_N', type=int, default=16, help="Number of augmentations")
    parser.add_argument('-K', '--sel_K', type=int, default=1, help="Number of images to select for inference")
    parser.add_argument('-bc', '--bias_c', type=float, default=None, help="Bias coefficient")
    parser.add_argument('-ct', '--conf_tresh', type=float, default=1, help="Confidence selection treshold")
    parser.add_argument('-o', '--optim', type=str, choices=['SGD', 'AdamW'], help="Optimizer")
    parser.add_argument('--lr', type=float, help="Optimizer learning rate")
    parser.add_argument('--prior_N', type=int, default=16, help="Norm adaptation factor")
    # Logs
    parser.add_argument('--wandb', action='store_true', default=False, help="Log on W&B")
    args = parser.parse_args()
    
    print("--- MEMO ---")
    
    # GPU
    DEVICE = get_device()
    print(f"Using device: {DEVICE}\n")

    # Norm adaptation
    assert not args.use_batch_stats or args.adapt
    norm_adaptation = args.adapt
    norm_adaptation_tag = "+NA" if norm_adaptation else "" 

    # Confidence treshold
    assert args.conf_tresh > 0 and args.conf_tresh <= 1
    assert args.sel_K <= int(args.aug_N * args.conf_tresh)
    conf_tresh_tag = "+CT" if args.conf_tresh < 1 else ""

    # Hyperparams
    hyperparams = {
        "aug": args.aug_type,
        "N": args.aug_N,
        "heuristic": 'TK',
        "sel_K": args.sel_K,
        "bias_c": args.bias_c,
        "conf_tresh": args.conf_tresh,
        "optim": args.optim,
        "lr": args.lr,
        "prior_N": args.prior_N,
        "use_batch_stats": args.use_batch_stats
    }
    print(f"Hyperparams: {hyperparams}")
    # Augmentation
    A = []
    augmentations_hyperparam = {"crop_size": 224, "antialias": True, "brightness": 0.5, "hue": 0.3}
    A.extend(get_augmentation(args.aug_type, augmentations_hyperparam))
    print(f"Augmentations pool: {A}\n")
    # Norm adaptation factor
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
            wandb.init(project=project_name, name=("TOPKMEMO_" + norm_adaptation_tag + conf_tresh_tag + "_ResNet50"), config=hyperparams)
        test_resnet50(hyperparams, A, norm_adaptation, DEVICE, wb)
    elif model_name == "ViT-B/16":
        if wb:
            wandb.init(project=project_name, name=("TOPKMEMO_" + conf_tresh_tag + "_ViT-B/16"), config=hyperparams)
        test_vit_b_16(hyperparams, A, DEVICE, wb)