"""
Explainable AI: Model-Agnostic Occlusion Sensitivity
Systematically perturbs image patches with a sliding window and measures output score degradation.
Directly tests: "Does covering this region remove the model's suspicion of forgery?"
"""

import torch
import numpy as np

class OcclusionExplainer:
    """Computes model-agnostic occlusion sensitivity map."""
    
    def __init__(self, model, patch_size=(32, 32), stride=(16, 16), fill_value=0.0):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.fill_value = fill_value
        
    def attribute(self, input_tensor, target_class=1):
        """
        Computes spatial occlusion importance map.
        Args:
            input_tensor: (1, 3, H, W) normalized tensor
            target_class: 1 for forged, 0 for authentic
        Returns:
            occlusion_map: (H, W) numpy array in [0.0, 1.0]
        """
        self.model.eval()
        device = input_tensor.device
        _, _, H, W = input_tensor.shape
        ph, pw = self.patch_size
        sh, sw = self.stride
        
        with torch.no_grad():
            orig_out = self.model(input_tensor)
            if isinstance(orig_out, dict):
                orig_score = orig_out.get("cls_prob", torch.sigmoid(orig_out.get("cls_logits", orig_out.get("logits")))).item()
            else:
                orig_score = torch.sigmoid(orig_out).item()
                
            if target_class == 0:
                orig_score = 1.0 - orig_score
                
        # Generate grid positions
        y_steps = range(0, H - ph + 1, sh)
        x_steps = range(0, W - pw + 1, sw)
        
        heatmap = np.zeros((H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)
        
        # Batch occluded inputs for fast execution
        batch_inputs = []
        batch_coords = []
        batch_size = 16
        
        with torch.no_grad():
            for y in y_steps:
                for x in x_steps:
                    occluded = input_tensor.clone()
                    occluded[:, :, y:y+ph, x:x+pw] = self.fill_value
                    batch_inputs.append(occluded)
                    batch_coords.append((y, x))
                    
                    if len(batch_inputs) == batch_size:
                        batch_tensor = torch.cat(batch_inputs, dim=0)
                        out = self.model(batch_tensor)
                        if isinstance(out, dict):
                            scores = out.get("cls_prob", torch.sigmoid(out.get("cls_logits", out.get("logits")))).squeeze().cpu().numpy()
                        else:
                            scores = torch.sigmoid(out).squeeze().cpu().numpy()
                            
                        if target_class == 0:
                            scores = 1.0 - scores
                        scores = np.atleast_1d(scores)
                            
                        for score, (by, bx) in zip(scores, batch_coords):
                            # Importance is drop in forgery score when occluded
                            delta = max(0.0, orig_score - float(score))
                            heatmap[by:by+ph, bx:bx+pw] += delta
                            counts[by:by+ph, bx:bx+pw] += 1.0
                            
                        batch_inputs = []
                        batch_coords = []
                        
            if batch_inputs:
                batch_tensor = torch.cat(batch_inputs, dim=0)
                out = self.model(batch_tensor)
                if isinstance(out, dict):
                    scores = out.get("cls_prob", torch.sigmoid(out.get("cls_logits", out.get("logits")))).squeeze().cpu().numpy()
                else:
                    scores = torch.sigmoid(out).squeeze().cpu().numpy()
                if target_class == 0:
                    scores = 1.0 - scores
                scores = np.atleast_1d(scores)
                for score, (by, bx) in zip(scores, batch_coords):
                    delta = max(0.0, orig_score - float(score))
                    heatmap[by:by+ph, bx:bx+pw] += delta
                    counts[by:by+ph, bx:bx+pw] += 1.0
                    
        # Average overlaps
        valid = counts > 0
        heatmap[valid] /= counts[valid]
        
        # Min-max normalization
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap)
            
        return heatmap
