"""
Explainable AI: Example-Based Explanations & Latent Space Retrieval
Answers: "Which known training documents does the model consider most similar to this document?"
Retrieves nearest authentic and forged exemplars using cosine similarity on deep latent embeddings.
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger("example_retrieval")

class ExampleBasedExplainer:
    """Manages an indexed database of training document embeddings for similarity explanations."""
    
    def __init__(self, database_embeddings=None, metadata_list=None):
        """
        database_embeddings: (N, D) numpy array of normalized deep feature embeddings
        metadata_list: list of dicts with keys: image_path, label, fraud_type, master_id
        """
        self.embeddings = database_embeddings
        self.metadata = metadata_list or []
        
    def index_dataset(self, model, dataloader, device="cpu"):
        """Extracts and indexes penultimate embeddings for an entire dataset split."""
        model.eval()
        embeddings = []
        meta_list = []
        
        with torch.no_grad():
            for imgs, masks, labels, metas in dataloader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                emb = outputs.get("embedding").cpu().numpy()
                # L2 normalize
                norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-7
                emb_norm = emb / norms
                
                embeddings.append(emb_norm)
                
                batch_size = len(labels)
                for i in range(batch_size):
                    meta_list.append({
                        "image_name": metas["image_name"][i],
                        "image_path": metas["image_path"][i],
                        "label": "forged" if labels[i].item() == 1.0 else "genuine",
                        "fraud_type": metas["fraud_type"][i],
                        "master_id": metas["master_id"][i]
                    })
                    
        self.embeddings = np.vstack(embeddings)
        self.metadata = meta_list
        logger.info(f"Indexed {len(self.metadata)} documents in latent embedding database.")
        
    def find_nearest_neighbors(self, query_embedding, k=3, exclude_master_id=None):
        """
        Retrieves top k nearest authentic and top k nearest forged exemplars.
        query_embedding: (D,) or (1, D)
        """
        q = query_embedding.reshape(1, -1)
        q_norm = q / (np.linalg.norm(q) + 1e-7)
        
        sims = cosine_similarity(q_norm, self.embeddings)[0]
        
        authentic_matches = []
        forged_matches = []
        
        # Sort indices by descending similarity
        ranked_indices = np.argsort(sims)[::-1]
        
        for idx in ranked_indices:
            rec = self.metadata[idx]
            # Prevent master document self-retrieval
            if exclude_master_id and rec["master_id"] == exclude_master_id:
                continue
                
            sim_score = float(sims[idx])
            match_info = {
                "image_name": rec["image_name"],
                "image_path": rec["image_path"],
                "label": rec["label"],
                "fraud_type": rec["fraud_type"],
                "master_id": rec["master_id"],
                "similarity_score": round(sim_score, 4)
            }
            
            if rec["label"] == "genuine" and len(authentic_matches) < k:
                authentic_matches.append(match_info)
            elif rec["label"] == "forged" and len(forged_matches) < k:
                forged_matches.append(match_info)
                
            if len(authentic_matches) >= k and len(forged_matches) >= k:
                break
                
        return {
            "nearest_authentic": authentic_matches,
            "nearest_forged": forged_matches
        }
