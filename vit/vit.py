from copy import deepcopy
import os
import logging
import argparse
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib # for tkinter error

matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import wandb
import evaluate
import time

from transformers import (
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from transformers.integrations import WandbCallback
from peft import get_peft_model, LoraConfig
from huggingface_hub import login
import timm
from transformers.modeling_outputs import SequenceClassifierOutput

from data.data_loader import create_data_loader

FULL_FINETUNE = False

class CustomTrainingArguments(TrainingArguments):
    def __init__(self, solar_quantize: str = 'none', *args, **kwargs):
        super().__init__(*args, **kwargs)

class PCAWeightApproximator:
    def __init__(self, W):
        self.W = W

    def svd_approximation(self, delta_W, n_components):
        device = delta_W.device
        W_cpu = self.W.detach()
        U, S, V = torch.svd(W_cpu)

        U_k = U[:, :n_components]
        V_k = V[:, :n_components]
        S_k = S[:n_components]

        approximation = torch.zeros_like(delta_W, device=device)

        for i in range(n_components):
            basis = S_k[i] * torch.ger(U_k[:, i], V_k[:, i]).to(device)
            alpha = torch.sum(delta_W * basis) / torch.sum(basis ** 2)
            approximation += alpha * basis

        return approximation


class FederatedLearningConfig:
    def __init__(
            self,
            foundation_model: str = "google/vit-base-patch16-224-in21k",
            dataset: str = "CIFAR10",
            batch_size: int = 64,
            num_epochs: int = 10,
            learning_rate: float = 2e-3,
            lora_rank: int = 8,
            num_clients: int = 5,
            target_layers: List[str] = ["key"]
    ):
        self.foundation_model = foundation_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.lora_rank = lora_rank
        self.num_clients = num_clients
        self.target_layers = target_layers

class SubspaceAnalyzer:
    @staticmethod
    def calculate_subspace_similarity(A: torch.Tensor, B: torch.Tensor, max_i: int, max_j: int) -> Dict[
        Tuple[int, int], torch.Tensor]:
        similarities = {}
        U_A = torch.linalg.svd(A)[0]
        U_B = torch.linalg.svd(B)[0]

        for i in range(1, max_i + 1):
            for j in range(1, max_j + 1):
                norm_value = torch.norm(U_A[:, :i].T @ U_B[:, :j], p='fro') ** 2
                similarity = norm_value / min(i, j)
                similarities[(i, j)] = similarity
        return similarities

    @staticmethod
    def calculate_eigenvector_similarity(A: torch.Tensor, B: torch.Tensor, max_i: int, max_j: int) -> Dict[
        Tuple[int, int], torch.Tensor]:
        similarities = {}
        U_A = torch.linalg.svd(A)[0]
        U_B = torch.linalg.svd(B)[0]

        for i in range(1, max_i + 1):
            for j in range(1, max_j + 1):
                similarity = torch.norm(U_A[:, i - 1:i].T @ U_B[:, j - 1:j], p='fro') ** 2
                similarities[(i, j)] = similarity
        return similarities

    @staticmethod
    def eigenvector_top_similar(A: torch.Tensor, B: torch.Tensor, N: int = 5000)  -> Dict[
        Tuple[int, int], torch.Tensor]:
        """Calculate normalized eigenvector similarity."""
        threshold = 1e-3
        A = F.normalize(A, p=2, dim=1)
        B = F.normalize(B, p=2, dim=0) # this make cos sim like dot product
        similarities = A @ B
        abs_sim = torch.abs(similarities)
        rank = A.size(0)


        # Separate positive and negative similarities
        pos_mask = similarities > 0
        neg_mask = similarities < 0
        # Normalize
        pos_prob = (abs_sim * pos_mask).float()
        neg_prob = (abs_sim * neg_mask).float()
        pos_prob /= pos_prob.sum() + 1e-8  # Avoid division by zero
        neg_prob /= neg_prob.sum() + 1e-8

        selected_pairs = {}

        for _ in range(N):
            if torch.rand(1).item() < 0.5:
                prob_dist = pos_prob
                mask = pos_mask
            else:
                prob_dist = neg_prob
                mask = neg_mask

            sampled_indices = torch.multinomial(prob_dist.flatten(), rank, replacement=True)

            sampled_i, sampled_j = torch.div(sampled_indices, B.shape[1], rounding_mode='floor'), sampled_indices % \
                                                                                                  B.shape[1]

            selected_pairs[_] = [(i.item(), j.item(), similarities[i, j].item()) for i, j in zip(sampled_i, sampled_j)]

        return selected_pairs

    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )


class CustomWandbCallback(WandbCallback):
    def __init__(self, lora_rank, target_layers, target_layer_num, pruned_model, masked_model, n_components=10, is_timm_model=False):
        super().__init__()
        self.wandb_logged = False
        self.lora_rank = lora_rank
        self.target_layers = target_layers
        self.target_layer_num = target_layer_num
        self.subspace_analyzer = SubspaceAnalyzer()
        self.n_components = n_components
        self.pca_approximator = PCAWeightApproximator(n_components)
        self.original_weights = {}
        self.pruned_model = pruned_model
        self.masked_model = masked_model
        self.is_timm_model = is_timm_model

    def _approximate_delta_w_with_pca(self, delta_w, n_components=768):
        pca = PCA(n_components=n_components)
        delta_w_pca = pca.fit_transform(
            delta_w.cpu().numpy().astype(np.float64))  # to decrease Floating-Point Precision error
        delta_w_reconstructed = torch.tensor(
            pca.inverse_transform(delta_w_pca), dtype=delta_w.dtype, device=delta_w.device  # =torch.float32
        )
        return delta_w_reconstructed

    def _compute_metrics(self, model, eval_dataset):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_dataset:
                if hasattr(model, "forward_features"):
                    logits = model(batch["pixel_values"])
                    predictions = logits.argmax(dim=-1)
                else:
                    outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1)

                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        return {'accuracy': correct / total}

    def _extract_layer_weights(self, model, target_layer):
        if hasattr(self, 'is_timm_model') and self.is_timm_model:
            print(f"[INFO] Skipping _extract_layer_weights for timm model.")
            return None
        attention_layer = model.vit.encoder.layer[self.target_layer_num].attention.attention
        layer_mapping = {
            "value": attention_layer.value,
            "key": attention_layer.key,
            "query": attention_layer.query
            # "dense":
        }

        layer = layer_mapping.get(target_layer)
        return {
            "w_value": layer.weight.clone().detach().cpu(),
            "w_A": layer.lora_A.default.weight.clone().detach().cpu(),
            "w_B": layer.lora_B.default.weight.clone().detach().cpu()
        }

    def frobenius_norm_projection(self, delta_A, A_optimized, r):
        if not isinstance(delta_A, torch.Tensor):
            delta_A = torch.tensor(delta_A, dtype=torch.float32)
        if not isinstance(A_optimized, torch.Tensor):
            A_optimized = torch.tensor(A_optimized, dtype=torch.float32)

        U, S, Vt = torch.linalg.svd(delta_A, full_matrices=False)

        U_r = U[:, :r]
        Vt_r = Vt[:r, :]

        A_proj = (U_r.T @ A_optimized @ Vt_r.T)

        frob_norm_proj = torch.norm(A_proj, p='fro').item()

        return frob_norm_proj

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        eval_logs = [log for log in state.log_history if "eval_loss" in log]
        if eval_logs and not self.wandb_logged:
            self.wandb_logged = True
            model = kwargs.get('model', None)

            target_layer = list(self.target_layers)[0]
            weights = self._extract_layer_weights(model, target_layer)

            if weights is None:
                print("[INFO] Skipping evaluation analysis for timm model.")
                return
            A_lora = weights['w_A']
            B_lora = weights['w_B']


            target_layer = list(self.target_layers)[0]
            masked_weights = self._extract_layer_weights(self.masked_model, target_layer)
            A_masked = masked_weights['w_A']
            B_masked = masked_weights['w_B']

            pruned_weights = self._extract_layer_weights(self.pruned_model, target_layer)
            A_pruned = pruned_weights['w_A']
            B_pruned = pruned_weights['w_B']


            similarity_norm_lora_mask = self.frobenius_norm_projection(A_lora, A_masked, r=len(A_lora))
            similarity_norm_lora_prune = self.frobenius_norm_projection(A_lora, A_pruned, r=len(A_lora))

            wandb.log({
                'similarity_norm_lora_mask': similarity_norm_lora_mask,
                'similarity_norm_lora_prune': similarity_norm_lora_prune,
            })

            delta_w = torch.matmul(weights['w_B'], weights['w_A'])
            w_updated = weights['w_value'] + delta_w

            # in here approximate delta w
            delta_w_pca_approx = self._approximate_delta_w_with_pca(delta_w)
            w_updated_pca = weights['w_value'] + delta_w_pca_approx

            reconstruction_error = torch.norm(delta_w - delta_w_pca_approx, p='fro') / torch.norm(delta_w, p='fro')
            wandb.log({
                'pca_reconstruction_error': reconstruction_error.item(),
                'frobenius_norm_pca': torch.norm(w_updated_pca, p='fro').item()
            })

            frobenius_norm = torch.norm(w_updated, p='fro')
            wandb.log({
                'frobenius_norm': frobenius_norm.item()
            })


            eigenvector_similarities = self.subspace_analyzer.calculate_eigenvector_similarity(
                weights['w_value'], delta_w,
                max_i=768, max_j=min(768, self.lora_rank)
            )

            similarities = self.subspace_analyzer.calculate_subspace_similarity(
                weights['w_value'], delta_w,
                max_i=768, max_j=min(768, self.lora_rank)
            )

            self._plot_similarity_heatmap(similarities, 'subspace')
            self._plot_similarity_heatmap(eigenvector_similarities, 'eigenvector')

        self.wandb_logged = False

    def _plot_similarity_heatmap(self, similarity_dict, similarity_type):

        i_values = sorted(set(i for i, j in similarity_dict.keys()))
        j_values = sorted(set(j for i, j in similarity_dict.keys()))
        similarity_matrix = np.zeros((len(i_values), len(j_values)))

        for (i, j), similarity in similarity_dict.items():
            similarity_matrix[i_values.index(i), j_values.index(j)] = similarity.item()

        # Compute dynamic color scaling range
        min_val = np.min(similarity_matrix)
        max_val = np.max(similarity_matrix)

        plt.figure(figsize=(4.5, 8))#, dpi=200)
        im = plt.imshow(similarity_matrix, cmap='plasma', aspect='auto', vmin=min_val, vmax=max_val) # cividis, plasma
        plt.colorbar(im)
        x_tick_positions = [j_values.index(j) for j in j_values if j in [1, 2, 3, 4]]
        plt.xticks(
            ticks=x_tick_positions,
            labels=[str(j) for j in [1, 2, 3, 4]],
            fontsize=20
        )
        plt.yticks(
            ticks=[i for i in range(0, len(i_values), 300)],
            labels=[str(i_values[i]) for i in range(0, len(i_values), 300)],
            fontsize=20
        )

        for x in x_tick_positions:
            plt.axvline(x=x, color='white', linestyle='--', linewidth=1.2, alpha=0.6)

        plt.xlabel('j', fontsize=20)
        plt.ylabel('i', fontsize=20)
        plt.title('Subspace Similarity', fontsize=22)
        plt.title('Query', fontsize=22)

        # Log to wandb and close
        wandb.log({f"{similarity_type}_similarity_heatmap": wandb.Image(plt)})
        plt.close()


class ModifiedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.n_components = kwargs.pop('n_components', 10)
        self.retain_params = kwargs.pop('retain_params', 10)
        self.num_random_basis = kwargs.pop('num_random_basis', 1000)
        self.num_random_basis_each_vector = kwargs.pop('num_random_basis_each_vector', 20)
        self.random_bases_scaling = kwargs.pop('random_bases_scaling', 1)
        self.useSimilarVectorsFromFoundation = kwargs.pop('useSimilarVectorsFromFoundation', True)
        self.is_timm_model = kwargs.pop('is_timm_model', False)
        self.pruned_model = kwargs.pop('pruned_model', None)
        self.masked_model = kwargs.pop('masked_model', None)
        self.full_finetune = kwargs.pop("full_finetune", False)
        self.evaluation_done = False
        self.solar_quantize = kwargs.pop("solar_quantize", False)
        super().__init__(*args, **kwargs)
        self.model_original = self.model

        if not self.full_finetune:
            for idx, callback in enumerate(self.callback_handler.callbacks):
                if isinstance(callback, WandbCallback):
                    self.callback_handler.callbacks[idx] = CustomWandbCallback(
                        lora_rank=self.model.peft_config['default'].r,
                        target_layers=self.model.peft_config['default'].target_modules,
                        target_layer_num=0,
                        pruned_model=self.pruned_model,
                        masked_model=self.masked_model,
                        n_components=self.n_components,
                        is_timm_model=self.is_timm_model,

                    )


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "timm" in self.args.output_dir or hasattr(model, "forward_features"):
            labels = inputs["labels"]
            logits = model(inputs["pixel_values"])
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return (loss, SequenceClassifierOutput(logits=logits)) if return_outputs else loss
        else:
            # Standard HuggingFace models
            return super().compute_loss(model, inputs, return_outputs=return_outputs)


    def masking_coeffiecints_random_basis(self, model, retain_params=10):

        updated_model = deepcopy(model)
        total_lora_elements = 0
        ifSVDpurturbation = False
        noProjection = False
        perturbSingulars = True
        useSimilarVectorsFromFoundation = self.useSimilarVectorsFromFoundation

        for name, module in updated_model.named_modules():
            if hasattr(module, 'weight') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                W = module.weight
                A = module.lora_A.default.weight
                B = module.lora_B.default.weight
                # Handle fused qkv projection (in timm ViT)
                if "qkv" in name:
                    hidden_size = W.shape[0] // 3  # infer from weight shape
                    W_timm = deepcopy(W)  # keep full qkv weight copy
                    B_timm = deepcopy(B)  # keep full qkv LoRA B matrix copy

                    # Extract query (first third)
                    W = W[:hidden_size, :]
                    B = B[:hidden_size, :]

                N = self.num_random_basis
                retain_params_num = int(retain_params * N) # check this part later
                retain_params_num = retain_params_num if retain_params_num < N else N

                # Compute SVD of W
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                V = Vt.T  # Right singular vectors (768x768)


                A_proj = A @ V
                B_proj = U.T @ B


                N = self.num_random_basis
                vector_dim = V.size(1)
                if perturbSingulars:
                    epsilon = self.random_bases_scaling
                    num_selected_vectors = A_proj.shape[0]  # Number of singular vectors to select
                    M_A = torch.zeros(N, A_proj.shape[0], A_proj.shape[1], device=A_proj.device)
                    if useSimilarVectorsFromFoundation:
                        all_indices = SubspaceAnalyzer.eigenvector_top_similar(
                            A_proj, V, N
                        )
                    for i in range(N):
                        if not useSimilarVectorsFromFoundation:
                            indices = torch.randperm(V.size(1))[:num_selected_vectors]  # Randomly select indices
                        else:
                            indices_i = all_indices[i]
                            indices_i = [item[1] for item in indices_i]
                            indices = torch.tensor(indices_i)
                        V_selected = V[:, indices]  # Select columns from V (right singular vectors)

                        noise = torch.randn_like(V_selected)  # Random noise of the same size as V_selected
                        noise_normalized = noise / torch.norm(noise, p=2, dim=0, keepdim=True)
                        V_perturbed = V_selected + epsilon * noise_normalized

                        M_A[i] = V_perturbed.T  # Transpose to match the desired size

                else:
                    M_A = torch.randn(N, A_proj.shape[0], A_proj.shape[1], device=A_proj.device)# @ Vt
                if ifSVDpurturbation: # can optimize this part
                    dot_products = abs(A_proj @ V) # (rows of Vt or cols of V are singular vectors)
                    top_k_indices = torch.topk(dot_products, k=self.num_random_basis_each_vector, dim=1).indices # here I have another hyper param
                    num_entries = top_k_indices.numel()
                    num_random_matrices = retain_params_num // num_entries
                    extra_entries = retain_params_num % num_entries
                    M_A = torch.zeros(retain_params_num, A_proj.size(0), A_proj.size(1), device=A_proj.device)

                    index = 0
                    for c in range(top_k_indices.size(1)):  # Iterate over columns
                        for r in range(top_k_indices.size(0)):  # Iterate over rows
                            vectorOfV = Vt[top_k_indices[r, c]] # corosponding singular vector of V (col of V)
                            num_samples = num_random_matrices + (
                                1 if extra_entries > 0 else 0)  # Distribute extra entries
                            if extra_entries > 0:
                                extra_entries -= 1

                            for _ in range(num_samples):
                                random_matrix = torch.randn(vector_dim, device=A_proj.device)# * scale_factor
                                M_A[index, r] = (1 - self.random_bases_scaling) * vectorOfV + self.random_bases_scaling * random_matrix
                                index += 1


                A_proj_flat = A_proj.flatten()
                M_A_flat = M_A.view(N, -1).T  # Shape: (m * n, N)

                if self.solar_quantize == '8bit':
                    alpha_tensor = torch.matmul(torch.pinverse(M_A_flat.float()), A_proj_flat.float())
                    alpha_q = torch.quantize_per_tensor(alpha_tensor.cpu(), scale=1e-5, zero_point=0,
                                                       dtype=torch.qint8)
                    alpha_tensor = alpha_q.dequantize().to(alpha_tensor.device)

                else:
                    lstsq_result = torch.linalg.lstsq(M_A_flat, A_proj_flat)
                    alpha_tensor = lstsq_result.solution  # Shape: (N,)

                if ifSVDpurturbation:
                    A_proj2 = M_A_flat @ alpha_tensor
                    A_proj = A_proj2.view(A_proj.shape)
                    A_approximated = A_proj @ Vt
                    torch.norm(A_approximated - A, p=2)


                else:


                    _, top_indices = torch.topk(alpha_tensor.abs().flatten(), retain_params_num)
                    mask = torch.zeros_like(alpha_tensor.flatten())
                    mask[top_indices] = 1
                    mask = mask.view_as(alpha_tensor)
                    alpha_tensor = alpha_tensor * mask

                    A_proj2 = M_A_flat @ alpha_tensor # .view(N)
                    A_proj = A_proj2.view(A_proj.shape)

                if perturbSingulars:
                    epsilon = epsilon  # for B
                    num_selected_vectors = B_proj.shape[1]  # Number of singular vectors to select
                    M_B = torch.zeros(N, B_proj.shape[0], B_proj.shape[1], device=B_proj.device)
                    if useSimilarVectorsFromFoundation:
                        all_indices = SubspaceAnalyzer.eigenvector_top_similar(
                            B_proj.T, U, N
                        )
                    for i in range(N):
                        if not useSimilarVectorsFromFoundation:
                            indices = torch.randperm(V.size(1))[:num_selected_vectors]  # Randomly select indices
                        else:
                            indices_i = all_indices[i]
                            indices_i = [item[1] for item in indices_i]
                            indices = torch.tensor(indices_i)
                        U_selected = U[indices, :]  # Select columns from V (right singular vectors)

                        noise = torch.randn_like(U_selected)  # Random noise of the same size as U_selected
                        noise_normalized = noise / torch.norm(noise, p=2, dim=0,
                                                              keepdim=True)  # Normalize noise column-wise
                        U_perturbed = U_selected + epsilon * noise_normalized

                        M_B[i] = U_perturbed.T

                else:
                    M_B = torch.randn(N, B_proj.shape[0], B_proj.shape[1], device=B_proj.device)
                if ifSVDpurturbation:  # can optimize this part
                    dot_products = abs(B_proj.T @ U)  # (rows of Vt or cols of V are singular vectors)
                    top_k_indices = torch.topk(dot_products, k=self.num_random_basis_each_vector,
                                               dim=1).indices  # here I have another hyper param
                    num_entries = top_k_indices.numel()
                    num_random_matrices = retain_params_num // num_entries
                    extra_entries = retain_params_num % num_entries
                    M_B = torch.zeros(retain_params_num, B_proj.size(0), B_proj.size(1), device=B_proj.device)

                    index = 0
                    for c in range(top_k_indices.size(1)):  # Iterate over columns
                        for r in range(top_k_indices.size(0)):  # Iterate over rows
                            vectorOfU = U.T[top_k_indices[r, c]]
                            num_samples = num_random_matrices + (
                                1 if extra_entries > 0 else 0)  # Distribute extra entries
                            if extra_entries > 0:
                                extra_entries -= 1

                            for _ in range(num_samples):
                                random_matrix = torch.randn(vector_dim, device=B_proj.device)# * scale_factor

                                M_B[index, :, r] = (1 - self.random_bases_scaling) * vectorOfU + self.random_bases_scaling * random_matrix
                                index += 1
                B_proj_flat = B_proj.flatten()  # Shape: (m * n,)
                M_B_flat = M_B.view(N, -1).T  # Shape: (m * n, N)

                if self.solar_quantize == '8bit':
                    # Quantize to 8-bit
                    beta_tensor = torch.matmul(torch.pinverse(M_B_flat.float()), B_proj_flat.float())
                    beta_q = torch.quantize_per_tensor(beta_tensor.cpu(), scale=1e-5, zero_point=0,
                                                       dtype=torch.qint8)
                    beta_tensor = beta_q.dequantize().to(beta_tensor.device)

                else:
                    lstsq_result_B = torch.linalg.lstsq(M_B_flat, B_proj_flat)
                    beta_tensor = lstsq_result_B.solution  # Shape: (N,)

                if not ifSVDpurturbation:
                    # beta_tensor = beta_tensor.view(N, 1, 1)

                    # Retain only the top retain_params_num elements in beta_tensor
                    _, top_indices_B = torch.topk(beta_tensor.abs().flatten(), retain_params_num)
                    mask_B = torch.zeros_like(beta_tensor.flatten())
                    mask_B[top_indices_B] = 1
                    mask_B = mask_B.view_as(beta_tensor)
                    beta_tensor = beta_tensor * mask_B

                    B_proj2 = M_B_flat @ beta_tensor #.view(N)  # Shape: (m * n,)
                    B_proj = B_proj2.view(B_proj.shape)  # Shape: (m, n)
                else:
                    B_proj2 = M_B_flat @ beta_tensor
                    B_proj = B_proj2.view(B_proj.shape)

                # Step 4: Approximate A and B
                A_approximated = A_proj @ Vt
                B_approximated = U @ B_proj
                if noProjection:
                    A_approximated = A_proj
                    B_approximated = B_proj

                if "qkv" in name:
                    B_approximated = torch.cat([B_approximated, B_timm[hidden_size:, :]], dim=0)

                module.lora_A.default.weight.data.copy_(A_approximated)
                module.lora_B.default.weight.data.copy_(B_approximated)
        if total_lora_elements == 0:  # Handle cases where there are no LoRA parameters
            return updated_model #, 0.0

        return updated_model

    def compute_basis_matrices(self, W, n_components):
        # PCA
        W_flat = W.view(W.size(0), -1).detach().cpu().numpy()  # Convert to NumPy for sklearn compatibility
        pca = PCA(n_components=n_components)
        pca.fit(W_flat)

        components = pca.components_  # Shape: (n_components, cols)
        singular_values = pca.singular_values_  # Shape: (n_components,)

        basis_matrices = []
        for i in range(n_components):
            component = torch.tensor(components[i], dtype=W.dtype, device=W.device).unsqueeze(0)  # Shape: (1, cols)
            basis_matrix = singular_values[i] * torch.matmul(component.T,
                                                             component)  # as later use scaler singular value can be ignored here # Shape: (rows, rows)
            basis_matrices.append(basis_matrix)



        return basis_matrices


    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                        metric_key_prefix="eval"):
        original_model = deepcopy(self.model)

        metrics_original = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix
        )

        start_train = time.time()

        masked_model_random_basis = self.masking_coeffiecints_random_basis(original_model,
                                                                           retain_params=self.retain_params)

        solar_train_time = time.time() - start_train
        print(f"solar train time:", solar_train_time)
        wandb.log({"timing/lora_train_time": solar_train_time})
        print(f"LoRA training time: {solar_train_time:.2f} seconds")

        self.model = masked_model_random_basis

        model_masking_random_basis_opt = super().evaluation_loop(
            dataloader,
            "SOLAR",
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix="SOLAR"
        )
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, CustomWandbCallback):
                callback.masked_model_random_basis = masked_model_random_basis


        self.model = self.model_original

        combined_metrics = metrics_original
        combined_metrics.metrics.update({
            f"{k}": v for k, v in model_masking_random_basis_opt.metrics.items()
        })

        # Log metrics to WandB
        if self.args.report_to == ["wandb"]:
            wandb.log({
                "eval/original_accuracy": metrics_original.metrics.get("eval_accuracy", 0),
                "eval/SOLAR_accuracy": model_masking_random_basis_opt.metrics.get(
                    "SOLAR_accuracy", 0),
                "eval/SOLAR_accuracy_diff": metrics_original.metrics.get("eval_accuracy", 0) -
                                                    model_masking_random_basis_opt.metrics.get(
                                                        "SOLAR_accuracy", 0),
                "eval/current_epoch": self.state.epoch
            })

        return combined_metrics



    def training_step(self, model, inputs, num_items_in_batch): # CPU
        """Override training step to log PCA metrics during training"""
        # Regular training step
        loss = super().training_step(model, inputs)

        # Periodically compute and log PCA metrics (e.g., every 100 steps)
        if self.state.global_step % 100 == 0:
            with torch.no_grad():
                if hasattr(model, "forward_features"):
                    logits = model(inputs["pixel_values"])
                else:
                    logits = model(**inputs).logits
                preds_original = logits.argmax(-1)
                correct_original = (preds_original == inputs["labels"]).float().mean()

            if self.args.report_to == ["wandb"]:
                wandb.log({
                    "train/original_accuracy": correct_original.item(),
                    "train/step": self.state.global_step,
                    "train/epoch": self.state.epoch
                })

        num_training_steps_per_epoch = self.state.max_steps // self.state.num_train_epochs
        if self.state.global_step % num_training_steps_per_epoch == 0:
            self.lora_gradients = {}
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    self.lora_gradients[name] = {
                        'lora_A_grad': module.lora_A.default.weight.grad.clone(),
                        'lora_B_grad': module.lora_B.default.weight.grad.clone(),
                    }

        return loss

    def _create_pca_approximated_model(self):
        if not hasattr(self, 'model_original'):
            self.model_original = self.model

        model_pca = deepcopy(self.model)

        for name, module in model_pca.named_modules():
            if any(target in name for target in self.model.peft_config['default'].target_modules):
                if hasattr(module, 'weight'):
                    W = module.weight
                else:
                    continue
                if hasattr(module, 'lora_A'):
                    A = module.lora_A.default.weight
                    B = module.lora_B.default.weight
                    delta_W = torch.matmul(B, A)
                    W_updated = W + delta_W

                    # Apply PCA
                    original_shape = W_updated.shape
                    W_2d = W_updated.detach().reshape(original_shape[0], -1).cpu().numpy()
                    pca = PCA(n_components=self.n_components)
                    W_pca = pca.fit_transform(W_2d)
                    W_reconstructed = pca.inverse_transform(W_pca)

                    W_reconstructed = torch.tensor(
                        W_reconstructed.reshape(original_shape),
                        dtype=W.dtype,
                        device=W.device
                    )
                    module.weight.data = W_reconstructed

                    # Zero out LoRA contributions since they're now included in W
                    module.lora_A.default.weight.data.zero_()
                    module.lora_B.default.weight.data.zero_()

        return model_pca


class FederatedLearningTrainer:
    def __init__(self, config: FederatedLearningConfig, args,  num_components: int = 10, retain_params: int = 10,
                 num_random_basis: int = 1000, num_random_basis_each_vector: int = 20,
                 random_bases_scaling: float = 0.1, useSimilarVectorsFromFoundation: bool = True) -> object:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.num_components = num_components
        self.retain_params = retain_params
        self.num_random_basis = num_random_basis
        self.num_random_basis_each_vector = num_random_basis_each_vector
        self.random_bases_scaling = random_bases_scaling
        self.useSimilarVectorsFromFoundation = useSimilarVectorsFromFoundation
        self.setup_logging()
        self.args = args
        self.pruned_model = None  # Initialize to None
        self.masked_model = None

    def setup_logging(self):
        """Configure logging and environment variables."""
        logging.basicConfig(level=logging.INFO)
        os.environ["WANDB_PROJECT"] = "Fast-PEFT"
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        torch.backends.cudnn.benchmark = True

    def load_model(self, label2id, id2label):
        if "facebook/vit-mae" in self.config.foundation_model:
            from transformers import ViTMAEModel
            model = ViTMAEModel.from_pretrained(self.config.foundation_model).to(self.device)
            model.classifier = torch.nn.Sequential(
                torch.nn.LayerNorm(model.config.hidden_size),
                torch.nn.Linear(model.config.hidden_size, len(label2id))
            ).to(self.device)
            model.config.label2id = label2id
            model.config.id2label = id2label
            model.config.num_labels = len(label2id)
            return model

        elif self.config.foundation_model.startswith("timm-"):
            # Strip timm: prefix and load model using timm
            timm_model_name = self.config.foundation_model.replace("timm-", "")
            model = timm.create_model(timm_model_name, pretrained=True, num_classes=len(label2id))
            model.to(self.device)
            model.label2id = label2id
            model.id2label = id2label
            model.num_labels = len(label2id)

            class DummyConfig(dict):
                _name_or_path = timm_model_name

                def to_dict(self):
                    return {"_name_or_path": self._name_or_path}

            model.config = DummyConfig()
            return model

        # Hugging Face vision transformer
        return AutoModelForImageClassification.from_pretrained(
            self.config.foundation_model,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        ).to(self.device)

    def setup_lora(self, model):
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=16,
            target_modules=self.config.target_layers,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        return get_peft_model(model, lora_config)

    def FLtrain(self):
        # Data loading
        train_loader, test_loader, train_ds, val_ds, label2id, id2label = create_data_loader(
            self.config.dataset,
            self.config.batch_size,
            self.config.foundation_model
        )

        # Model preparation
        foundation_model = self.load_model(label2id, id2label)
        SubspaceAnalyzer.print_trainable_parameters(foundation_model)
        if not FULL_FINETUNE:
            lora_model = self.setup_lora(foundation_model)
        else:
            for param in foundation_model.parameters():
                param.requires_grad = True  # Ensure full fine-tuning
            foundation_model.train()
            lora_model = foundation_model
        SubspaceAnalyzer.print_trainable_parameters(lora_model)

        # Training setup
        training_args = CustomTrainingArguments(
            output_dir=f"{self.config.foundation_model.split('/')[-1]}-lora",
            run_name=f"{self.config.foundation_model.split('/')[-1]}-lora",
            report_to="wandb",
            remove_unused_columns=False,
            # eval_strategy="no",
            save_strategy="no",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=self.config.batch_size,
            fp16=True,
            num_train_epochs=self.config.num_epochs,
            logging_steps=1,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=True,
            label_names=["labels"],
            ddp_find_unused_parameters=False,
            solar_quantize=self.args.solar_quantize,

        )

        self.pruned_model = deepcopy(lora_model)
        self.masked_model = deepcopy(lora_model)


        trainer = ModifiedTrainer(
            model=lora_model,
            pruned_model=self.pruned_model,
            masked_model=self.masked_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=self._compute_metrics,
            data_collator=self._collate_fn,
            n_components=self.num_components,
            retain_params=self.retain_params,
            num_random_basis=self.num_random_basis,
            num_random_basis_each_vector=self.num_random_basis_each_vector,
            random_bases_scaling=self.random_bases_scaling,
            useSimilarVectorsFromFoundation=self.useSimilarVectorsFromFoundation,
            is_timm_model=self.args.is_timm_model,
            full_finetune=False, #
            solar_quantize=self.args.solar_quantize,
            callbacks=[CustomWandbCallback(
                self.config.lora_rank,
                self.config.target_layers,
                target_layer_num=0,
                pruned_model=self.pruned_model,  # Now defined
                masked_model=self.masked_model,  # Now defined
            )],
        )

        # Training
        train_results = trainer.train()
        trainer.evaluate()
        trainer.save_model('./fine_tuned_vit')

        if self.args.checkMatrixDim and not self.args.is_timm_model:

            print("\nExtracting w and Δw after fine-tuning...\n")
            w_value_updated = {}
            w_A_updated = {}
            w_B_updated = {}

            attention_layer = lora_model.vit.encoder.layer[0].attention.attention

            for target_layer in self.config.target_layers:
                layer_mapping = {
                    "value": attention_layer.value,
                    "key": attention_layer.key,
                    "query": attention_layer.query
                }

                layer = layer_mapping.get(target_layer)

                w_value_updated[target_layer] = layer.weight.clone().detach().cpu()
                w_A_updated[target_layer] = layer.lora_A.default.weight.clone().detach().cpu()
                w_B_updated[target_layer] = layer.lora_B.default.weight.clone().detach().cpu()

            print("Original w (Value matrix) shape:", w_value_updated[self.config.target_layers[0]].shape)
            print("Delta w A (LoRA A matrix) shape:", w_A_updated[self.config.target_layers[0]].shape)
            print("Delta w B (LoRA B matrix) shape:", w_B_updated[self.config.target_layers[0]].shape)

            delta_w = torch.matmul(w_B_updated[self.config.target_layers[0]], w_A_updated[self.config.target_layers[0]])

            logging.info("Fine-tuning completed and model saved.")

            return train_results, {
                'w_value_updated': w_value_updated,
                'w_A_updated': w_A_updated,
                'w_B_updated': w_B_updated,
                'delta_w': delta_w
            }
        return



    @staticmethod
    def _compute_metrics(eval_pred):
        """Compute accuracy metrics."""
        metric = evaluate.load("accuracy")
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    @staticmethod
    def _collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        if "fine_label" in examples[0]:
            labels = torch.tensor([example["fine_label"] for example in examples])
        else:
            labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="Train a LoRA model")
    # jonathancui/oxford-pets - pets, tanganke/sun397, bentrevett/caltech-ucsd-birds-200-2011
    parser.add_argument('--dataset', type=str, default='CIFAR10') # CIFAR10, CIFAR100, food101, zh-plus/tiny-imagenet, imagenet-1k
    parser.add_argument('--batch_size', type=int, default=128) # 128, for 10 sample use 16
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=2e-3)
    parser.add_argument('--foundation_model', type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--target_layers', nargs='+', default=["value"])  # dense, "key", query
    parser.add_argument('--num_components', type=int, default=200,
                        help='Number of components for PCA/SVD approximation')
    parser.add_argument('--retain_params', type=float, default= 0.1, # 0.25 : 768
                        help='Percentage of retaining parameters (percentage)')
    parser.add_argument('--checkMatrixDim', type=bool, default=True, help='matrix Dimension')
    parser.add_argument('--num_random_basis', type=int, default=1000, help='Value of N for random basis')
    parser.add_argument('--num_random_basis_each_vector', type=int, default=10, help='# of random bases for each singular vector')
    parser.add_argument('--random_bases_scaling', type=float, default=1, help='scaling of added random matrix')
    parser.add_argument('--solar_quantize', type=str, default='none', choices=['none', '8bit'],
                        help='quantization mode for SOLAR weights (none/linear/log/4bit/8bit)')
    parser.add_argument('--useSimilarVectorsFromFoundation', type=bool, default=False, help='leverage vector similarity of foundation model')


    args = parser.parse_args()

    args.top_k_params = int(args.num_random_basis * args.retain_params)

    args.is_timm_model = "timm" in args.foundation_model.lower() # to handle model naming and order of AB in lora instead of BA

    # Login to Hugging Face
    login(token='hf_.....', add_to_git_credential=False) # update token here

    if "mae" in args.foundation_model.lower() or "timm-" in args.foundation_model.lower():
        print("Auto-detected MAE/timm model. Updating target_layers for LoRA injection.")
        args.target_layers = ["attn.qkv"] # "attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"

    # wandb
    wandb.init(
        project="SOLAR",
        name=f"Fast-PEFT_target{args.target_layers}_retain_params{args.retain_params}_target_layers{args.target_layers}_rank{args.lora_rank}",
        config=vars(args)
    )

    wandb.config.update({"top_k_params": args.top_k_params})

    config = FederatedLearningConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        foundation_model=args.foundation_model,
        lora_rank=args.lora_rank,
        target_layers=args.target_layers,

    )

    print("========== CONFIGURATION ==========")
    print(f"Dataset: {args.dataset}")
    print(f"Foundation Model: {args.foundation_model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Num Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"LoRA Rank: {args.lora_rank}")
    print(f"Target Layers: {args.target_layers}")
    print(f"Num Random Basis: {args.num_random_basis}")
    print(f"Top K Retain Params: {args.retain_params * args.num_random_basis}")
    print("====================================")

    FLtrainer = FederatedLearningTrainer(config, args, num_components=args.num_components, retain_params=args.retain_params,
                                         num_random_basis = args.num_random_basis, num_random_basis_each_vector = args.num_random_basis_each_vector,
                                         random_bases_scaling = args.random_bases_scaling,
                                         useSimilarVectorsFromFoundation = args.useSimilarVectorsFromFoundation)
    start_train = time.time()
    results = FLtrainer.FLtrain()
    lora_train_time = time.time() - start_train
    print(f"lora train time:", lora_train_time)
    wandb.log({"timing/lora_train_time": lora_train_time})
    print(f"LoRA training time: {lora_train_time:.2f} seconds")


    wandb.finish()


if __name__ == "__main__":
    main()
