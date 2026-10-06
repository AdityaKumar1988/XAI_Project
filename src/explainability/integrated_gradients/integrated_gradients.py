"""
Explainable AI: Integrated Gradients for Pixel-Level Attribution
Axiomatic attribution satisfying Completeness, Implementation Invariance, and Linearity.
Computes path integral of gradients along linear interpolation between baseline and input.
"""

import torch
import numpy as np

class IntegratedGradientsExplainer:
    """Computes Integrated Gradients attribution for PyTorch forgery detection models."""
    
    def __init__(self, model, steps=30):
        self.model = model
        self.steps = steps
        
    def attribute(self, input_tensor, baseline=None, target_class=1):
        """
        Computes Integrated Gradients.
        Args:
            input_tensor: (1, 3, H, W) normalized tensor
            baseline: (1, 3, H, W) reference tensor (default: zeros)
            target_class: 1 for forged, 0 for authentic
        Returns:
            attributions: (H, W) normalized attribution magnitude map in [0.0, 1.0]
        """
        self.model.eval()
        device = input_tensor.device
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor).to(device)
            
        alphas = torch.linspace(0.0, 1.0, self.steps, device=device)
        
        # Linear interpolation path
        # (steps, 3, H, W)
        interpolated = baseline + alphas.view(-1, 1, 1, 1) * (input_tensor - baseline)
        interpolated.requires_grad_(True)
        
        # Forward pass on interpolated inputs in mini-batches to avoid VRAM peaks
        grads = []
        batch_size = 10
        for i in range(0, self.steps, batch_size):
            batch = interpolated[i:i+batch_size].clone().detach().requires_grad_(True)
            output = self.model(batch)
            if isinstance(output, dict):
                logits = output.get("cls_logits", output.get("logits"))
            else:
                logits = output
                
            scores = logits.squeeze() if target_class == 1 else -logits.squeeze()
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            grad = torch.autograd.grad(scores.sum(), batch)[0]
            grads.append(grad)
            
        all_grads = torch.cat(grads, dim=0) # (steps, 3, H, W)
        avg_grads = torch.mean(all_grads, dim=0, keepdim=True) # (1, 3, H, W)
        
        delta = input_tensor - baseline
        ig = delta * avg_grads # (1, 3, H, W)
        
        # Aggregate across color channels
        ig_mag = torch.sum(torch.abs(ig), dim=1).squeeze().detach().cpu().numpy() # (H, W)
        
        # Min-max normalization
        ig_min, ig_max = ig_mag.min(), ig_mag.max()
        if ig_max > ig_min:
            ig_norm = (ig_mag - ig_min) / (ig_max - ig_min)
        else:
            ig_norm = np.zeros_like(ig_mag)
            
        return ig_norm
