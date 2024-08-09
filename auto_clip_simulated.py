#!/usr/local/bin/python3
# the paper "AutoCLIP: Auto-tuning Zero-Shot Classifiers for Vision-Language Models" accepted at TMLR.
# Copyright (c) 2024 Robert Bosch GmbH
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Jan Hendrik Metzen, Piyapat Saranrittichai
# -*- coding: utf-8 -*-
""" Evaluates AutoCLIP on simulated CLIP data corresponding to Section 5 of https://openreview.net/forum?id=gVNyEVKjqf """

import numpy as np
from scipy.optimize import bisect
from scipy.special import softmax

__import__("matplotlib").use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns

import torch


# To ease understanding of the code, we annotate tensors with their corresponding dimensions:
#  - k: Number of embedding dimension
#  - d: Number of prompt templates (descriptor sets) - these would come for instance from WaffleCLIP or DCLIP
#  - i: Number of instances (data points)
#  - g: Number of ground-truth classses
#  - c: Number of classification logits
# Note that g and c have the same value, but the semantics of the respective tensor dimension is different:
#   g indicates to which class an instance actually belongs and c indicates the pre-softmax
#   logits the simulated zero-shot classifier assigns to this class


def simulate_clip(
    n_dims, n_classes, n_descriptors, instance_multiple, noise_scale, entanglement
):
    """Simulate CLIP image and text encoder assuming certain distributional properties.

    Parameters controlling the distributions:
      - n_dims: Number of dimensions of embedding space
      - n_classes: Number of classes
      - n_descriptors: Number of prompt templates (class descriptors)
      - instance_multiple: Number of instance per class
      - noise_scale: Instance noise scale (how much image embeddings of the same class vary)
      - entanglement: How strongly class and prompt template embeddings are entangled under the simulated CLIP text encoder
    """
    ### Sample class descriptors (gdk)
    # Unobserved class means in text embedding space
    class_means = np.random.normal(loc=0.0, scale=1.0, size=(n_classes, n_dims))  # (gk)
    # Unobserved prompt template means in text embedding space
    prompt_means = np.random.normal(
        loc=0.0, scale=1.0, size=(n_descriptors, n_dims)
    )  # (dk)
    # Entanglement of class and prompt template embeddings
    class_prompt_coupling = np.random.normal(
        loc=0.0, scale=1.0, size=(n_classes, n_descriptors, n_dims)
    )  # (dk)
    # Cumpte class descriptors as weighted combination of disentangled and fully entangled embedding
    class_descriptors = (1 - entanglement) * (
        class_means[:, None] + prompt_means[None]
    ) + entanglement * class_prompt_coupling  # (gdk)

    ### Sample image embeddings (ick)
    # For each instance, draw a corresponding prompt template (shared over all classes)
    selected_indices = np.random.choice(
        class_descriptors.shape[1], size=(instance_multiple)
    )  # (i)
    # Set the image embedding mean to the class descriptor of the selected prompt template
    image_descriptor_means = np.swapaxes(
        class_descriptors[:, selected_indices], 0, 1
    )  # (ick)
    # Add instance noise and add to image_descriptor means
    instance_variation = np.random.normal(
        loc=0.0, scale=noise_scale, size=(instance_multiple, n_classes, n_dims)
    )  # (ick)
    instances = image_descriptor_means + instance_variation  # (ick)

    # L2 normalization of embeddings
    class_descriptors /= np.linalg.norm(
        class_descriptors, ord=2, axis=-1, keepdims=True
    )  # (gdk)
    instances /= np.linalg.norm(instances, ord=2, axis=-1, keepdims=True)  # (ick)

    # Compute cosine similarities
    cosine_similarities = np.einsum(
        "gdk,ick->icgd", class_descriptors, instances
    )  # (icgd)

    return cosine_similarities


def softmax_aggregation(
    cosine_similarities, instance_multiple, n_classes, n_descriptors, beta
):
    ### Softmax aggregation from Section A.3 from https://openreview.net/forum?id=gVNyEVKjqf
    # Initialize weights uniformly (weights used for weighted class descriptor averaging)
    weights_logits = torch.nn.parameter.Parameter(
        torch.zeros((instance_multiple, n_classes, n_descriptors))
    )
    weights_logits.requires_grad = True
    weights = weights_logits.softmax(dim=2)  #  (icd)

    # Determine target entropy (compare Section 3.4 of https://openreview.net/forum?id=gVNyEVKjqf)
    base_entropy = (
        -(weights * torch.log2(weights)).sum(-1).detach().cpu().numpy()
    )  # (ic)
    target_entropy_step = base_entropy * beta

    # Select softmax temperatures based on line search to match target_entropy
    # (not batched and in numpy, could be optimized to decrease overall runtime)
    temperatures = torch.ones(
        (
            instance_multiple,
            n_classes,
        )
    )
    for i in range(instance_multiple):
        for j in range(n_classes):

            def f(temperature, verbose=False):
                # Compute difference of actual and target entropy
                weights_ = softmax(
                    temperature * cosine_similarities.mean(2)[i, j], axis=-1
                )
                entropy = -(weights_ * np.log2(weights_ + 1e-10)).sum(0)
                return entropy - target_entropy_step[i, j]

            temperatures[i, j] = bisect(
                f=f,
                a=0.01,
                b=10000,
                maxiter=100,
                xtol=1e-2,
                rtol=1e-2,
            )

    # Compute final class logits
    weights = torch.softmax(
        temperatures[:, :, None] * cosine_similarities.mean(2), dim=2
    )
    logits = torch.sum(
        weights[:, :, None] * torch.from_numpy(cosine_similarities), -1
    )  # (icg)
    return logits.detach().numpy()


def autoclip_aggregation(
    cosine_similarities, instance_multiple, n_classes, n_descriptors, beta
):
    ### AutoCLIP aggregation from Section 3.2 from https://openreview.net/forum?id=gVNyEVKjqf
    # Initialize weights uniformly (weights used for weighted class descriptor averaging)
    weights_logits = torch.nn.parameter.Parameter(
        torch.zeros((instance_multiple, n_classes, n_descriptors))
    )
    weights_logits.requires_grad = True
    weights = weights_logits.softmax(dim=2)  #  (icd)

    # Compute log-sum-exp of class logits and its gradient wrt. the weight's logits
    logits = torch.sum(
        weights[:, None] * torch.from_numpy(cosine_similarities), -1
    )  # (icg)
    lse = torch.logsumexp(logits, axis=2).sum()
    grad_weights_logits = torch.autograd.grad(lse, weights_logits)[0]

    # Determine target entropy (compare Section 3.4 of https://openreview.net/forum?id=gVNyEVKjqf)
    base_entropy = (
        -(weights * torch.log2(weights)).sum(-1).detach().cpu().numpy()
    )  # (ic)
    target_entropy_step = base_entropy * beta

    # Select step sizes based on line search to match target_entropy
    # (not batched and in numpy, could be optimized to decrease overall runtime)
    step_sizes = torch.ones(
        (
            instance_multiple,
            n_classes,
        )
    )
    grad_weights_logits_np = grad_weights_logits.detach().cpu().numpy()
    weights_logits_np = weights_logits.detach().cpu().numpy()
    for i in range(instance_multiple):
        for j in range(n_classes):

            def f(step_size, verbose=False):
                # Compute difference of actual and target entropy
                weights_logits_ = (
                    weights_logits_np[i, j] + step_size * grad_weights_logits_np[i, j]
                )
                weights_ = softmax(weights_logits_, axis=-1)
                entropy = -(weights_ * np.log2(weights_ + 1e-10)).sum(-1)
                return entropy - target_entropy_step[i, j]

            step_sizes[i, j] = bisect(
                f=f,
                a=0.0,
                b=1e7,
                maxiter=100,
                xtol=1e-2,
                rtol=1e-2,
            )

    # Do one step of gradient ascent on the weight's logits for the determined step size
    weights_logits.data += step_sizes[:, :, None] * grad_weights_logits
    weights = weights_logits.softmax(dim=2)  #  (icd)
    # Compute final class logits
    logits = torch.sum(
        weights[:, :, None] * torch.from_numpy(cosine_similarities), -1
    )  # (icg)
    return logits.detach().numpy()


def run_experiment(
    aggregation,
    n_dims,
    n_classes,
    n_descriptors,
    instance_multiple,
    noise_scale,
    entanglement,
    beta=0.85,
    n_repetitions=25,
):
    accuracies = []
    for seed in range(n_repetitions):
        np.random.seed(seed)

        # Simulate CLIP text and image encoders
        # Note: Actual CLIP modle could be "plugged-in" here to run AutoCLIP on actual
        #       zero-shot classification tasks
        cosine_similarities = simulate_clip(
            n_dims,
            n_classes,
            n_descriptors,
            instance_multiple,
            noise_scale,
            entanglement,
        )

        # Perform aggregation:
        if aggregation == "mean":
            class_logits = cosine_similarities.mean(-1)  # mean over descriptors (icg)
        elif aggregation == "max":
            class_logits = cosine_similarities.max(-1)  # max over descriptors (icg)
        elif aggregation == "softmax":
            class_logits = softmax_aggregation(
                cosine_similarities, instance_multiple, n_classes, n_descriptors, beta
            )  # (icg)
        elif aggregation == "AutoCLIP":
            class_logits = autoclip_aggregation(
                cosine_similarities, instance_multiple, n_classes, n_descriptors, beta
            )  # (icg)
        # Arg-max based classification
        classification = np.argmax(class_logits, 1)  # ig

        # Evaluate accuracy
        target = np.tile(np.arange(n_classes), (instance_multiple, 1))  # ig
        accuracies.append((classification == target).mean())

    return np.mean(accuracies)


if __name__ == "__main__":
    n_dims = 128
    n_classes = 5
    n_descriptors = 10
    instance_multiple = 200 // n_classes
    n_repetitions = 100

    entanglements = np.linspace(0, 1, 11)

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    rcParams = {
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }
    plt.rcParams.update(rcParams)

    fig_width_pt = 487.8225
    inches_per_pt = 1 / 72.27
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * 0.4
    fig_dim = (fig_width_in, fig_height_in)
    fig = plt.figure(figsize=fig_dim)

    for i, noise_scale in enumerate([2.5, 5.0, 10.0]):
        plt.subplot(1, 3, i + 1)
        for aggregation in ["mean", "max", "AutoCLIP", "softmax"]:
            print("Aggreagtion:", aggregation)
            mean_accuracies = []
            for entanglement in entanglements:
                mean_accuracy = run_experiment(
                    aggregation,
                    n_dims,
                    n_classes,
                    n_descriptors,
                    instance_multiple,
                    noise_scale,
                    entanglement,
                    n_repetitions=n_repetitions,
                )
                print("\t Entanglement: %.2f  Accuracy: %.2f" % (entanglement, mean_accuracy))
                mean_accuracies.append(mean_accuracy)

            plt.plot(entanglements, mean_accuracies, label=aggregation)

        plt.xlim(0, 1)
        # plt.ylim(0.5, 1.0)
        plt.xlabel("Entanglement")
        if i == 0:
            plt.legend()
        plt.ylabel("Accuracy")
        plt.title("Instance Noise: %d" % int(noise_scale))
    plt.tight_layout()
    plt.savefig("results_toy_data.pdf")
    plt.show()
