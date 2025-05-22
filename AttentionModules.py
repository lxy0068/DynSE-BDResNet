import torch
import torch.nn as nn
from typing import Optional


class ChannelAttention(nn.Module):
    """Channel Attention Module with adaptive kernel size.

    Implements efficient channel attention with adaptive kernel sizing for feature maps of different scales.

    Args:
        in_channels (int): Number of input feature map channels
        reduction_ratio (int): Feature compression ratio (default: 16)
        kernel_size (int): Convolution kernel size (auto-adapted to odd numbers, default: adaptive)
        gate_activation (nn.Module): Gate activation function (default: Sigmoid)

    Raises:
        ValueError: When reduction_ratio <= 0

    Shape:
        - Input: (N, C, H, W)
        - Output: Same as input
    """

    def __init__(
            self,
            in_channels: int,
            reduction_ratio: int = 16,
            kernel_size: Optional[int] = None,
            gate_activation: nn.Module = nn.Sigmoid()
    ):
        super().__init__()

        if reduction_ratio <= 0:
            raise ValueError(f"Reduction ratio must be >0, got {reduction_ratio}")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Automatically compute optimal kernel size
        if kernel_size is None:
            kernel_size = self._optimal_kernel_size(in_channels)

        # Dynamic intermediate feature dimension
        hidden_dim = max(1, in_channels // reduction_ratio)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=kernel_size // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size, padding=kernel_size // 2, bias=False)
        )

        self.gate = gate_activation

    def _optimal_kernel_size(self, channels: int) -> int:
        """Dynamically compute optimal kernel size"""
        return 3 if channels % 3 == 0 else 5 if channels % 5 == 0 else 7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing feature recalibration"""
        avg_out = self.conv(self.avg_pool(x))
        max_out = self.conv(self.max_pool(x))
        return x * self.gate(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module enhancing feature response in significant regions"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)

    Composite attention module combining channel and spatial attention for enhanced feature selection.

    Args:
        in_channels (int): Number of input channels
        reduction_ratio (int): Channel compression ratio (default: 16)
        spatial_kernel (int): Spatial attention kernel size (default: 7)

    Shape:
        - Input: (N, C, H, W)
        - Output: Same as input
    """

    def __init__(
            self,
            in_channels: int,
            reduction_ratio: int = 16,
            spatial_kernel: int = 7
    ):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction_ratio)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential application of channel and spatial attention"""
        x = self.channel(x)
        return self.spatial(x)


class SEBlock(nn.Module):
    """Enhanced Squeeze-and-Excitation Block

    Improved SE block with dynamic feature compression and hybrid pooling strategy.

    Args:
        in_channels (int): Number of input channels
        reduction_ratio (int): Feature compression ratio (default: 16)
        use_max_pool (bool): Enable max pooling for enhanced feature extraction (default: True)
    """

    def __init__(
            self,
            in_channels: int,
            reduction_ratio: int = 16,
            use_max_pool: bool = True
    ):
        super().__init__()
        self.use_max_pool = use_max_pool
        hidden_dim = max(1, in_channels // reduction_ratio)

        # Hybrid pooling layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if use_max_pool:
            self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Dual-path feature transformation
        self.fc = nn.Sequential(
            nn.Linear(2 * in_channels if use_max_pool else in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()

        # Hybrid pooling
        avg = self.avg_pool(x).view(batch_size, channels)
        if self.use_max_pool:
            max_pool = self.max_pool(x).view(batch_size, channels)
            combined = torch.cat([avg, max_pool], dim=1)
        else:
            combined = avg

        # Feature recalibration
        weights = self.fc(combined).view(batch_size, channels, 1, 1)
        return x * weights.expand_as(x)