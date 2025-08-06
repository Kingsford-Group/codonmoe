"""
CodonMoE: Adaptive Mixture of Codon Reformative Experts.

This module contains the main components of the CodonMoE model, including
the LayerNorm, MixtureOfExperts, CodonMoE, and mRNAModel classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Custom Layer Normalization implementation."""

    def __init__(self, features: int, eps: float = 1e-6):
        """
        Initialize the LayerNorm module.

        Args:
            features (int): The number of features in the input.
            eps (float): A small value added for numerical stability.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MixtureOfExperts(nn.Module):
    """Mixture of Experts implementation."""

    def __init__(self, input_dim: int, num_experts: int = 4):
        """
        Initialize the MixtureOfExperts module.

        Args:
            input_dim (int): The input dimension.
            num_experts (int): The number of expert networks.
        """
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, input_dim // 3)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the Mixture of Experts.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying MoE.
        """
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        gates = F.softmax(self.gate(x), dim=-1)
        return (expert_outputs * gates.unsqueeze(-2)).sum(dim=-1)


class CodonMoE(nn.Module):
    """Codon Mixture of Experts model."""

    def __init__(self, input_dim: int, num_experts: int = 4, dropout_rate: float = 0.1):
        """
        Initialize the CodonMoE module.

        Args:
            input_dim (int): The input dimension.
            num_experts (int): The number of expert networks.
            dropout_rate (float): The dropout rate for regularization.
        """
        super().__init__()
        self.d_model = input_dim
        self.moe = MixtureOfExperts(input_dim * 3, num_experts)
        self.flatten = nn.Flatten()
        self.fc1 = None
        self.fc2 = nn.Linear(input_dim, 1)
        self.layernorm1 = LayerNorm(input_dim)
        self.layernorm2 = LayerNorm(input_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the CodonMoE model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying CodonMoE.
        """
        batch_size, seq_len, _ = x.shape
        num_features = x.shape[1]

        if self.fc1 is None or self.fc1.in_features != self.d_model * num_features:
            self.fc1 = nn.Linear(self.d_model * num_features, self.d_model).to(x.device)
        #this depends on the backbone model
        seq_len = seq_len - 2
        y = x[:, :seq_len, :]

        y_codons = y.view(batch_size, seq_len // 3, 3 * self.d_model)
        y_moe = self.moe(y_codons)
        codon_info_full = y_moe.repeat(1, 3, 1).reshape(batch_size, seq_len, self.d_model)
        y_output = y + codon_info_full

        x_new = x.clone()
        x_new[:, :seq_len, :] = y_output
        x_new = self.layernorm1(x_new)
        x_new = self.activation(x_new)
        x_new = self.dropout(x_new)

        x_new = self.flatten(x_new)
        x_new = self.fc1(x_new)
        x_new = self.layernorm2(x_new)
        x_new = self.activation(x_new)
        x_new = self.dropout(x_new)
        output = self.fc2(x_new)
        return output


class mRNAModel(nn.Module):
    """mRNA model combining a base model with CodonMoE."""

    def __init__(self, base_model: nn.Module, codon_moe: CodonMoE):
        """
        Initialize the mRNAModel.

        Args:
            base_model (nn.Module): The base model to be used.
            codon_moe (CodonMoE): The CodonMoE model.
        """
        super().__init__()
        self.base_model = base_model
        self.codon_moe = codon_moe

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the mRNAModel.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.

        Returns:
            torch.Tensor: Output tensor after applying the full model.
        """
        outputs = self.base_model(input_ids)
        codon_output = self.codon_moe(outputs)
        return codon_output