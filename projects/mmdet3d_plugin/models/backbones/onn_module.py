import torch
import time
import torch.nn as nn
import numpy as np
from torchvision import transforms


def padding(x, nd):
    n, c, dd = x.shape
    # x = x.reshape(n, int(np.sqrt(c)), int(np.sqrt(c)))
    M = nd
    padded_x = torch.zeros((n, M, M), device=x.device, dtype=x.dtype)
    start_idx = (M - x.shape[-1]) // 2
    padded_x[:,start_idx:start_idx+x.shape[-1], start_idx:start_idx+x.shape[-1]] = x
    return padded_x

def depadding(x, ni):
    M = ni
    start_idx = (x.shape[-1] - M) // 2
    x = x[:,start_idx:start_idx+M, start_idx:start_idx+M]
    return x


def batch_padding(x, nd):
    b, n, c, dd = x.shape
    # x = x.reshape(n, int(np.sqrt(c)), int(np.sqrt(c)))
    M = nd
    padded_x = torch.zeros((b, n, M, M), device=x.device, dtype=x.dtype)
    start_idx_x = (M - x.shape[-1]) // 2
    start_idx_y = (M - x.shape[-2]) // 2
    padded_x[..., start_idx_y:start_idx_y+x.shape[-2], start_idx_x:start_idx_x+x.shape[-1]] = x
    return padded_x

def batch_depadding(x, ori_shape):
    _, _, N, M = ori_shape
    start_idx_x = (x.shape[-1] - M) // 2
    start_idx_y = (x.shape[-2] - N) // 2
    x = x[...,start_idx_y:start_idx_y+N, start_idx_x:start_idx_x+M]
    return x

class ASM_propagation(nn.Module):
    def __init__(self, L, lmbda, z):
        super(ASM_propagation, self).__init__()
        self.L = L
        self.lmbda = lmbda
        self.z = z
    
    def forward(self, u1):
        device = u1.device
        tmp = torch.complex(torch.tensor(0., device=device), torch.tensor(1., device=device))
        bs, c1, c2, M, N = u1.shape
        dx = self.L / M
        k = 2 * np.pi / self.lmbda

        fx = torch.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / self.L, M, dtype=torch.float32).to(device)
        FX, FY = torch.meshgrid(fx, fx, indexing='ij')
        mask = torch.where((self.lmbda**2 * FX**2 + self.lmbda**2 * FY**2) > 1, 0, 1)
        # mask = mask.unsqueeze(0).unsqueeze(0).repeat(bs, c1, c2, 1, 1)
        
        H = torch.exp(tmp * 2*torch.pi * self.z / self.lmbda * torch.sqrt((1 - self.lmbda**2 * FX**2 - self.lmbda**2 * FY**2))).to(device)
        # H = H.unsqueeze(0).unsqueeze(0).repeat(bs, c1, c2, 1, 1)

        H = torch.fft.fftshift(H, dim=(-2,-1)) * mask
        H = H.unsqueeze(0).unsqueeze(0).repeat(bs, c1, c2, 1, 1)
        u1 = torch.fft.fft2(torch.fft.fftshift(u1, dim=(-2,-1))).to(device)
        u1 = H * u1
        u1 = torch.fft.ifftshift(torch.fft.ifft2(u1), dim=(-2,-1))
        return u1

class phase_modulation(nn.Module):
    def __init__(self, M, N, out_channels, in_channels):
        super(phase_modulation, self).__init__()
        self.phase_values = nn.Parameter(torch.ones((out_channels, in_channels, N, N)))
        self.M = M
        self.N = N
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_weights()
    
    def init_weights(self):
        nn.init.uniform_(self.phase_values, a=0, b=2)

    def forward(self, input_tensor):
        device = input_tensor.device
        bs = input_tensor.shape[0]
        # Create a zero matrix of size M x M
        tmp = torch.complex(torch.tensor(0., device=device), torch.tensor(1., device=device))
        padded_phase_values = torch.zeros((bs, self.out_channels, self.in_channels, self.M, self.M), device=device)
        start_idx = (self.M - self.N) // 2
        # Place the N x N phase_values at the center of the M x M matrix
        padded_phase_values[..., start_idx:start_idx+self.N, start_idx:start_idx+self.N] = self.phase_values
        padded_phase_values = torch.exp(tmp * 2 * torch.pi * padded_phase_values)
        input_tensor = input_tensor * padded_phase_values
        return input_tensor

class OpticalLayer(nn.Module):
    def __init__(self, M, L, lmbda, z1, z2, N, ni, layersCount, out_channels, in_channels, use_bias=False, use_abs=False):
        self.M = M
        self.ni = ni
        super(OpticalLayer, self).__init__()
        layers = []
        layers.append(ASM_propagation(L, lmbda, z1))
        for _ in range(layersCount):
            layers.append(phase_modulation(M, N, out_channels, in_channels))
            layers.append(ASM_propagation(L, lmbda, z2))
        self.optical_layers = nn.Sequential(*layers)

        self.out_channels = out_channels
        self.weights = nn.Parameter(torch.ones(out_channels, in_channels))
        self.use_bias = use_bias
        self.use_abs = use_abs
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weights, mean=1.0, std=0.1)  # 使用正态分布初始化
        if self.use_bias:
            nn.init.constant_(self.bias, -0.1)

    def forward(self, x):
        # first padding the x to make it M x M
        bs = len(x)
        ori_shape = x.shape
        x = batch_padding(x, self.M)
        x = x[:, None].repeat(1, self.out_channels, 1, 1, 1)
        x = self.optical_layers(x)
        x = batch_depadding(x, ori_shape)
        if self.use_abs:
            x = torch.abs(x)
        if self.use_bias:
            bias = self.bias[None, :, None, None, None].repeat(bs, 1, *x.shape[2:])
            x += bias
        weights = self.weights[None, :, :, None, None].repeat(bs, 1, 1, 1, 1)
        x = x * weights
        x = x.sum(dim=2).float()
        return x

class ONNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, M, L, lmbda, z1, z2, N, ni, layersCount, use_bias=False, use_abs=False):
        super().__init__()
        if use_bias:
            assert use_abs

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.optical_layer = OpticalLayer(M, L, lmbda, z1, z2, N, ni, layersCount, out_channels, in_channels, use_bias, use_abs)

    def forward(self, x):
        outputs = self.optical_layer(x)
        return outputs


if __name__ == '__main__':
    config = {"conv1_out_channels": 2,
            "conv2_out_channels": 4, 
            "lmbda": 640e-9,
            "dx": 1.75e-6,
            "nd": 200,
            "d1": 100e-6,
            "d2": 100e-6,
            "N": 96,
            "layersCount": 20
    }
    L = config['nd'] * config['dx']
    net = ONNLayer(3, 8, 
                    config['nd'], 
                    L, 
                    config['lmbda'], 
                    config['d1'], 
                    config['d2'], 
                    config['N'], 
                    28, 
                    config['layersCount']).cuda()
    
    # input = torch.ones((2, 3, 24, 24)).cuda()
    output = net(input)
