"""
Explainable AI: PyTorch Grad-CAM and Grad-CAM++ Implementations
Provides:
- GradCAM: First-order gradient-weighted class activation mapping
- GradCAMPlusPlus: Higher-order pixel-weighted gradient activation mapping for multi-instance forgery
- Overlay visualization generator
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAMExplainer:
    """Computes Grad-CAM and Grad-CAM++ attributions on PyTorch vision models."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate_gradcam(self, input_tensor, target_class=1):
        """
        Computes standard Grad-CAM heatmap.
        Args:
            input_tensor: (1, 3, H, W) normalized tensor
            target_class: 1 for forged, 0 for authentic
        Returns:
            heatmap: (H, W) numpy array in [0.0, 1.0]
        """
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if isinstance(output, dict):
            logit = output.get("cls_logits", output.get("logits"))
        else:
            logit = output
            
        score = logit if target_class == 1 else -logit
        score.backward(retain_graph=True)
        
        gradients = self.gradients.detach() # (1, C, h, w)
        activations = self.activations.detach() # (1, C, h, w)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True) # (1, C, 1, 1)
        
        cam = torch.sum(weights * activations, dim=1, keepdim=True) # (1, 1, h, w)
        cam = F.relu(cam)
        
        # Resize to input dimensions
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

    def generate_gradcam_plus_plus(self, input_tensor, target_class=1):
        """
        Computes Grad-CAM++ heatmap with higher-order gradient weighting.
        Particularly effective for multi-target text forgery.
        """
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if isinstance(output, dict):
            logit = output.get("cls_logits", output.get("logits"))
        else:
            logit = output
            
        score = logit if target_class == 1 else -logit
        score.backward(retain_graph=True)
        
        gradients = self.gradients.detach() # (1, C, h, w)
        activations = self.activations.detach() # (1, C, h, w)
        
        # Compute 2nd and 3rd derivatives
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        eps = 1e-7
        alpha_denom = 2.0 * grad_2 + sum_activations * grad_3 + eps
        alpha = grad_2 / alpha_denom
        
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)
        
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

    @staticmethod
    def overlay_heatmap(image_np, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlays heatmap on original uint8 RGB image.
        Returns: (H, W, 3) uint8 image.
        """
        h, w = image_np.shape[:2]
        heatmap_resized = cv2.resize((heatmap * 255).astype(np.uint8), (w, h))
        colored = cv2.applyColorMap(heatmap_resized, colormap)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image_np, 1.0 - alpha, colored_rgb, alpha, 0)
        return overlay
