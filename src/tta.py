## -- TTA inference --

import torch
import torch.nn as nn
from torchvision.transforms import v2
import random
import numpy as np
import copy

# TTA methods #

# RMEMO #

# Inference via RMEMO
def RMEMO_inference(model, x, aug_x, conf_tresh, heuristic, sel_K, bias_c, optimizer, device):

    # Setup model
    model.eval()
    model.to(device)
    # Sanity check
    optimizer.zero_grad()
 
    # Apply the pre-trained model to the sampled augmentations of input x
    # (N, c, H, W) -> (N, |Y|)
    y_aug = model(aug_x)
    # Compute Log Normalized probabilities by softmax :- log(p(y|x))
    # (N, |Y|) -> (N, |Y|)
    y_log_prob = y_aug.log_softmax(dim=1)
    # Filter samples by confidence in the prediction
    # (N, |Y|) -> (conf_N, |Y|)
    filt_idxs, topk_idxs = filter_samples_by_confidence_plus_tk(y_log_prob, conf_tresh, sel_K)
    y_log_prob = y_log_prob[filt_idxs,:]

    # Gradient descend step to minimze the entropy of the marginal distribution (l_me)
    # (conf_N, |Y|) -> (1)
    loss = marginal_entropy_loss(y_log_prob)
    # Backprop
    loss.backward()
    # Update parameters
    optimizer.step()

    # Select images for inference (C)
    if heuristic == 'KL':
        selected_imgs_idxs = get_kl_closer_imgs_idxs(y_log_prob, sel_K)
    elif heuristic == 'TK':
        selected_imgs_idxs = topk_idxs
    elif heuristic == 'SR':
        selected_imgs_idxs = get_sr_closer_imgs_idxs(y_log_prob, sel_K)
    else:
        raise NotImplementedError(f"Unknown Heuristic: {heuristic}")
    # (conf_N, c, H, W) -> (sel_K, c, H, W)
    selected_imgs = aug_x[selected_imgs_idxs,:,:,:]

    # Inference from updated model
    # (sel_K, c, H, W) -> (1, |Y|)
    with torch.no_grad():
        if bias_c:
            output = biased_inference(model, x, selected_imgs, bias_c, device)
        else:
            output = multi_augmentation_inference(model, selected_imgs, 1, device)   

    return output

# Select images for final inference via KL heuristic
def get_kl_closer_imgs_idxs(y_log_prob, sel_K):

    # Compute marginal distribution (p_m) over the sampled augmentations
    # (N, |Y|) -> (1, |Y|)
    marg_y_log_prob = marginal_log_probability(y_log_prob).unsqueeze(0)
    N = torch.tensor(y_log_prob.size()[0])
    # (1, |Y|) -> (N, |Y|)
    rep_marg_y_log_prob = marg_y_log_prob.repeat(N, 1)

    # Compute KL divergence from each image to marginal distribution (p_m)
    # (N, |Y|) -> (N)
    kl_divs = torch.nn.functional.kl_div(y_log_prob, rep_marg_y_log_prob, reduction='none', log_target=True)
    kl_divs = kl_divs.sum(dim=1)

    # Get top K images idxs
    _, idxs = torch.sort(kl_divs, dim=0, descending=False, stable=True)
    idxs = idxs[0:sel_K]
    
    return idxs

# Select images for final inference via SR heuristic
def get_sr_closer_imgs_idxs(y_log_prob, sel_K):

    # Compute marginal distribution (p_m) over the sampled augmentations
    # (N, |Y|) -> (1, |Y|)
    marg_y_log_prob = marginal_log_probability(y_log_prob).unsqueeze(0)
    N = torch.tensor(y_log_prob.size()[0])
    # (1, |Y|) -> (N, |Y|)
    rep_marg_y_log_prob = marg_y_log_prob.repeat(N, 1)

    # Compute KL divergence for each image to marginal distribution (p_m)
    # (N, |Y|) -> (N)
    kl_divs = torch.nn.functional.kl_div(y_log_prob, rep_marg_y_log_prob, reduction='none', log_target=True)
    kl_divs = kl_divs.sum(dim=1)

    # Compute entropy for each augmented sample
    # (N, |Y|) -> (N)
    samples_entropy = -1 * torch.sum(y_log_prob * torch.exp(y_log_prob), dim=1)

    # Score ranking
    # (N) -> (N)
    _, kl_idxs = torch.sort(kl_divs, dim=0, descending=False, stable=True)
    _, ent_idxs = torch.sort(samples_entropy, dim=0, descending=False, stable=True)
    kl_idxs, ent_idxs = kl_idxs.tolist(), ent_idxs.tolist()
    scores = {i: kl_idxs.index(i)+ent_idxs.index(i) for i in range(0,N)}
    idxs = sorted(scores, key=scores.get)
    
    # Get Top K confident images idxs for inference
    idxs = idxs[0:sel_K]
    
    return idxs

# Filter samples by entropy tresholding, via percentille of discrete distribution
# Return also indexes of Top K samples
# NOTE! - Requires Log space probabilities
# (N, |Y|) -> (conf_N, |Y|)
def filter_samples_by_confidence_plus_tk(y_log_prob, conf_tresh, sel_K):

    # Compute entropy for each augmented sample
    # (N, |Y|) -> (N)
    samples_entropy = -1 * torch.sum(y_log_prob * torch.exp(y_log_prob), dim=1)
    # Round percentile
    rounded_conf_tresh = int(y_log_prob.size()[0] * conf_tresh)
    # Filter samples by entropy tresholding
    # (N, |Y|) -> (rounded_conf_tresh, |Y|)
    _, idxs = torch.sort(samples_entropy, dim=0, descending=False, stable=True)
    filt_idxs = idxs[0:rounded_conf_tresh]

    # Get Top K confident images idxs for inference
    topk_idxs = idxs[0:sel_K]

    return filt_idxs, topk_idxs

# MEMO #

# Inference via MEMO
def MEMO_inference(model, x, aug_x, conf_tresh, optimizer, device):

    # Setup model
    model.eval()
    model.to(device)
    # Sanity check
    optimizer.zero_grad()

    # Apply the pre-trained model to the sampled augmentations of input x
    # (N, c, H, W) -> (N, |Y|)
    y_aug = model(aug_x)
    # Compute Log Normalized probabilities by softmax :- log(p(y|x))
    # (N, |Y|) -> (N, |Y|)
    y_log_prob = y_aug.log_softmax(dim=1)
    # Filter samples by confidence in the prediction
    # (N, |Y|) -> (conf_N, |Y|)
    filt_idxs = filter_samples_by_confidence(y_log_prob, conf_tresh)
    y_log_prob = y_log_prob[filt_idxs,:]

    # Gradient descend step to minimze the entropy of the marginal distribution (l_me)
    # (conf_N, |Y|) -> (1)
    loss = marginal_entropy_loss(y_log_prob)
    # Backprop
    loss.backward()
    # Update parameters
    optimizer.step()

    # Inference from updated model
    # (1, c, H, W) -> (1, |Y|)
    with torch.no_grad():
        output = model(x.unsqueeze(0)).log_softmax(dim=1)

    return output

# Filter samples by entropy tresholding, via percentille of discrete distribution
# NOTE! - Requires Log space probabilities
# (N, |Y|) -> (conf_N, |Y|)
def filter_samples_by_confidence(y_log_prob, conf_tresh):

    # Compute entropy for each augmented sample
    # (N, |Y|) -> (N)
    samples_entropy = -1 * torch.sum(y_log_prob * torch.exp(y_log_prob), dim=1)
    # Round percentile
    rounded_conf_tresh = int(y_log_prob.size()[0] * conf_tresh)
    # Filter samples by entropy tresholding
    # (N, |Y|) -> (rounded_conf_tresh, |Y|)
    _, idxs = torch.sort(samples_entropy, dim=0, descending=False, stable=True)
    filt_idxs = idxs[0:rounded_conf_tresh]

    return filt_idxs

# Compute the entropy of marginal distrubution over a set of samples
# NOTE! - Requires Log space probabilities
# (N, |Y|) -> (1)
def marginal_entropy_loss(y_log_prob):

    # Marginal probability in Log space
    # (N, |Y|) -> (|Y|)
    marg_y_log_prob = marginal_log_probability(y_log_prob)
    # Marginal entropy
    # (|Y|) -> (1)
    entropy = -1 * torch.sum(marg_y_log_prob * torch.exp(marg_y_log_prob))

    return entropy

# Compute marginal probability in Log space
# NOTE! - Requires Log space probabilities
# (N, |Y|) -> (|Y|)
def marginal_log_probability(y_log_prob):

    # Marginal probabilities :- 1/N * sum(p(y|x))
    # Mean in Log space :- log(1) - log(N) + log(sum(p(y|x))),
    # where :- log(sum(p(y|x))) = log(sum(exp(log(p(y|x)))))
    # (N, |Y|) -> (|Y|)
    N = torch.tensor(y_log_prob.size()[0])
    marg_y_log_prob = -1 * (torch.log(N)) + torch.logsumexp(y_log_prob, dim=0)

    return marg_y_log_prob

# MA #-

# Inference via marginalization over the distribution of augmented images
def multi_augmentation_inference(model, aug_x, conf_tresh, device):

    # Setup model
    model.eval()
    model.to(device)
    
    # Apply the model to the sampled augmentations of input x
    # (N, c, H, W) -> (N, |Y|)
    y_aug = model(aug_x)
    # Compute Log Normalized probabilities by softmax :- log(p(y|x))
    # (N, |Y|) -> (N, |Y|)
    y_log_prob = y_aug.log_softmax(dim=1)
    # Filter samples by confidence in the prediction
    # (N, |Y|) -> (conf_N, |Y|)
    filt_idxs = filter_samples_by_confidence(y_log_prob, conf_tresh)
    y_log_prob = y_log_prob[filt_idxs,:]
    # Marginal probabilities
    # (N, |Y|) -> (1, |Y|)
    marg_y_log_prob = marginal_log_probability(y_log_prob)
    output = torch.exp(marg_y_log_prob).unsqueeze(0)

    return output

# BIASED #

# Biased inference by linear interpolation between x and aug_x predictions using bias_c coefficient 
def biased_inference(model, x, aug_x, bias_c, device):

    # Setup model
    model.eval()
    model.to(device)
    
    # Apply the model to the original image + sampled augmentations of input x
    # (1, c, H, W), (N, c, H, W) -> (N+1, c, H, W)
    biased_aug_x = torch.cat((x.unsqueeze(0), aug_x), dim=0) 
    # (N+1, c, H, W) -> (N+1, |Y|)
    y_aug = model(biased_aug_x)
    # Compute Log Normalized probabilities by softmax :- log(p(y|x))
    # (N+1, |Y|) -> (N+1, |Y|)
    y_log_prob = y_aug.log_softmax(dim=1)
    # Split prediction of original image from augmented images
    # (N+1, |Y|) -> (1, |Y|), (N, |Y|) 
    id_y_log_prob, aug_y_log_prob = y_log_prob[0,:].unsqueeze(0), y_log_prob[1:,]
    # Marginal probabilities of augmentations
    # (N, |Y|) -> (1, |Y|)
    aug_marg_y_log_prob = marginal_log_probability(aug_y_log_prob).unsqueeze(0)
    # Linear interpolation between predictions
    # (1, |Y|) -> (1, |Y|)
    biased_marg_y_log_prob = bias_c * id_y_log_prob + (1 - bias_c) * aug_marg_y_log_prob
    output = torch.exp(biased_marg_y_log_prob)

    return output

# NORM ADAPTATION #

# MEMO version of Batch Norm Adaptive computation
# NOTE: Inference ONLY, i.e. works for model.eval() NOT for model.train()
#       If keep_test_running is False, behave as simple norm adaptation
class MEMOAdaptiveBatchNorm(nn.Module):

    def __init__(self, pre_bn_layer, prior_N, keep_test_running):
        super().__init__()
        # Previous BN layer from pre-trained model
        self.pre_bn_layer = pre_bn_layer
        # Norm Adaptation hyperparameters
        self.prior_N = prior_N
        # Use batch stats 1st forward pass (v_test) for 2nd forward pass
        self.keep_test_running = keep_test_running
        self.test_batch_mean = None
        self.test_batch_var = None

    # Adapt normalization statistics
    def forward(self, input):

        # Sanity check model.eval only
        assert (not self.pre_bn_layer.training)
        # Torch required assertion
        self.pre_bn_layer._check_input_dim(input)

        # 1st forward pass (test batch mean and var None)
        if self.test_batch_mean is None and self.test_batch_var is None:

            # Compute test batch normalization statistics
            # (N, c, H, W) -> (c)
            curr_mean, curr_var = self.compute_norm_stats(input)
            # Save 1st forward batch stats
            if self.keep_test_running:
              self.test_batch_mean = curr_mean
              self.test_batch_var = curr_var

        # 2nd forward pass
        else:

            # If batch statistics are used for both 1st and 2nd forward pass
            if self.keep_test_running:
              curr_mean = self.test_batch_mean
              curr_var = self.test_batch_var
            # Otherwise compute norm stats from current input
            else:
              # (N, c, H, W) -> (c)
              curr_mean, curr_var = self.compute_norm_stats(input)

            # Consistency reset
            self.test_batch_mean = None
            self.test_batch_var = None

        # Modify statistics via linear combination of v_train and v_test
        # a * v_train + (1-a) * v_test
        alpha = self.prior_N / (self.prior_N + 1)
        running_mean = alpha * self.pre_bn_layer.running_mean + (1 - alpha) * curr_mean
        running_var = alpha * self.pre_bn_layer.running_var + (1 - alpha) * curr_var

        # Return adapted normalized statistics (layer output)
        # (N, c, H, W)
        return nn.functional.batch_norm(input, running_mean, running_var, self.pre_bn_layer.weight,
                                        self.pre_bn_layer.bias, False, 0, self.pre_bn_layer.eps)

    # HELPERS

    # Compute normalization statistics from a given batch/instance
    def compute_norm_stats(self, input):

        # Compute current batch statistics v_test[mean,var]
        N, c, H, W = input.shape
        # (N, c, H, W) -> (c, N, H, W)
        t_input = input.transpose(0, 1).detach()
        # Flatten out hidden units h_i per channel (c, h_i),
        # (c, N, H, W) -> (c, N*H*W)
        collapsed_input = t_input.contiguous().view(c, N*H*W)
        # Compute statistics (mean,var) over the channels
        # (c, N*H*W) -> (c)
        mean = collapsed_input.mean(dim=1)
        var = collapsed_input.var(dim=1, unbiased=False)

        return mean, var

# Change Normalization Layers (e.g. BatchNorm2d) behaviour in the model
# Allows to use MEMO batch statistics of 1st forward pass as stats for 2nd pass
class MEMOAdaptiveBatchNormModel(nn.Module):
    def __init__(self, model, prior_N, keep_test_running, copy_model=True):
        super().__init__()
        # Pre-trained model
        self.model = copy.deepcopy(model) if copy_model else model
        # Norm Adaptation
        self.prior_N = prior_N
        # Use batch stats 1st forward pass (v_test) for 2nd forward pass
        self.keep_test_running = keep_test_running
        # Override _BatchNorm Layers
        self.adapt_Batch_Norm_layers()

    def forward(self, x):
        return self.model(x)

    # HELPERS

    # Modify behaviour of _BatchNorm layers of the model with MEMOAdaptiveBN custom one
    def adapt_Batch_Norm_layers(self):

        # Find BN modules (layers)
        bn_modules = self.find_bn_modules(self.model)

        # Iterate over found BN modules
        for parent, name, new_module in bn_modules:
            # Substitute BN module with MEMOAdaptiveBN
            setattr(parent, name, new_module)

    # Tree-like search of Batch Norm modules in the model
    def find_bn_modules(self, parent):

        bn_modules = []
        # Iterate over module's childrens
        for name, child_module in parent.named_children():

            # If child module is BN layer
            if isinstance(child_module, torch.nn.modules.batchnorm._BatchNorm):
                # Compose new MEMOAdaptiveBN and add to the list of modules to substitute
                new_module = MEMOAdaptiveBatchNorm(child_module, self.prior_N, self.keep_test_running)
                bn_modules.append((parent, name, new_module))
            # Otherwise search recursively
            else:
                bn_modules.extend(self.find_bn_modules(child_module))

        return bn_modules

# OPTIMIZERS #

# Get fresh optimizer from parameters
def get_optimizer(model, name, hyperparam):

    # Check optimizer suppport
    assert name in ["SGD","AdamW"]
    # SGD
    if name == "SGD":
        return torch.optim.SGD(params=model.parameters(),**hyperparam)
    # AdamW
    if name == "AdamW":
        return torch.optim.AdamW(params=model.parameters(),**hyperparam)
    return None
    
# AUGMENTATIONS #

# Get augmentations for a given technique
def get_augmentation(name, hyperparam):

    # Check augmentations support
    assert name in ["AugMix","RandomResizedCrop","Mixture"] 
    
    # AugMix
    if name == "AugMix":
        return [v2.AugMix()]
    # RandomResizedCrops
    if name == "RandomResizedCrop":
        return [v2.RandomResizedCrop(size=(hyperparam["crop_size"],hyperparam["crop_size"]), antialias=hyperparam["antialias"])]
    # Mixture
    if name == "Mixture":
        return [v2.AugMix(), 
                v2.ColorJitter(brightness=hyperparam["brightness"], hue=hyperparam["hue"]), 
                v2.RandomResizedCrop(size=(hyperparam["crop_size"],hyperparam["crop_size"]), antialias=hyperparam["antialias"])]
        
    return None
