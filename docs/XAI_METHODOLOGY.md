# Explainable AI (XAI) Methodology & Theoretical Foundations

**Project:** Explainable AI System for Document Forgery Detection, Localization, and Decision Interpretation  
**Date:** October 2026  
**Document Reference:** `docs/XAI_METHODOLOGY.md`  

---

## 1. Why Explainable AI is Indispensable for Document Forensics

In financial audits, judicial proceedings, academic credential verification, and immigration inspection, deep learning systems cannot operate as inscrutable "black boxes." A binary prediction (e.g., `FORGED (94.2%)`) without verifiable justification is inadmissible in legal or administrative disputes.

Document forensics poses unique interpretability challenges:
1. **Microscopic Footprint:** Forgeries frequently alter fewer than 0.5% of total document pixels (e.g., changing a single digit in a GPA or a name string). Standard global attribution methods easily drown out these localized anomalies.
2. **Adversarial Saliency / Spurious Correlations:** Standard CNNs may predict "forged" not because they detected tampered ink boundaries, but because of background paper grain, scanner tilt, or institutional logos.
3. **Multi-Stakeholder Needs:** 
   - *Forensic Examiners* require pixel-level heatmaps and compression disparity metrics.
   - *Legal Audits* require contrastive counterfactuals ("What would flip this verdict?").
   - *Non-technical Users* require case-based exemplars ("Show me verified documents that look like this").

Therefore, our architecture implements an **integrated multi-modal XAI suite** addressing six fundamental forensic questions:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   THE SIX FORENSIC XAI QUESTIONS                       │
├─────────────────────────┬────────────────────────┬─────────────────────┤
│ Question                │ Explanation Type       │ Methodology         │
├─────────────────────────┼────────────────────────┼─────────────────────┤
│ 1. Is it authentic?     │ Decision Verdict       │ Calibrated MT-Model │
│ 2. WHY did it predict?  │ Visual Attribution     │ Grad-CAM / IG / Occ │
│ 3. WHERE is tampering?  │ Localization Mask      │ Decoder Head + BBox │
│ 4. WHICH features?      │ Tabular Attribution    │ TreeSHAP & LIME     │
│ 5. WHAT would change?   │ Counterfactual         │ Minimal Shift & ELA │
│ 6. WHICH are similar?   │ Exemplar Retrieval     │ Latent Embedding NN │
└─────────────────────────┴────────────────────────┴─────────────────────┘
```

---

## 2. Visual Explanation Techniques

### 2.1 Grad-CAM (Gradient-Weighted Class Activation Mapping)
Grad-CAM computes the gradient of the target class score $Y^c$ with respect to feature activation maps $A^k$ of the final convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial Y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

*Application in Document Forensics:* Identifies coarse spatial regions ($32 \times 32$ downsampled resolution) that contributed positively to the classification of tampering.

### 2.2 Grad-CAM++ (Higher-Order Pixel-Weighted Attribution)
Standard Grad-CAM struggles when a document contains **multiple disconnected tampering sites** (e.g., both GPA and attendance modified) because gradients are globally averaged across the spatial dimensions. Grad-CAM++ introduces pixel-specific weights $w_{i,j}^{k,c}$ derived from second and third derivatives:

$$w_{i,j}^{k,c} = \frac{\frac{\partial^2 Y^c}{(\partial A_{i,j}^k)^2}}{2 \frac{\partial^2 Y^c}{(\partial A_{i,j}^k)^2} + \sum_{a=1}^u \sum_{b=1}^v A_{a,b}^k \frac{\partial^3 Y^c}{(\partial A_{a,b}^k)^3} + \epsilon}$$

$$\alpha_k^c = \sum_{i=1}^u \sum_{j=1}^v w_{i,j}^{k,c} \cdot \text{ReLU}\left(\frac{\partial Y^c}{\partial A_{i,j}^k}\right)$$

*Result:* Generates tighter, multi-focal heatmaps that isolate individual spliced text glyphs without smearing.

### 2.3 Integrated Gradients (Axiomatic Path Attribution)
A major limitation of CAM methods is gradient saturation (where high-confidence activations produce near-zero gradients). Integrated Gradients satisfies two fundamental axioms:
1. **Completeness:** The sum of attributions equals the difference between model output at $x$ and baseline $x'$:
   $$\sum_i \text{IG}_i(x) = F(x) - F(x')$$
2. **Implementation Invariance:** Two functionally identical models yield identical attributions.

Integrated Gradients computes the straight-line path integral between neutral reference $x'$ (blank document paper) and input $x$:

$$\text{IG}_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

Approximated via Riemann summation across $m = 25$ steps:

$$\text{IG}_i^{\text{approx}}(x) = (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^m \frac{\partial F\left(x' + \frac{k}{m}(x - x')\right)}{\partial x_i}$$

### 2.4 Occlusion Sensitivity (Model-Agnostic Perturbation)
To verify whether the deep model is genuinely sensitive to tampering boundaries or simply relying on background contextual artifacts, a sliding square patch of size $P \times P$ occludes local regions with stride $S$:

$$\Delta S(u, v) = S_{\text{original}} - S_{\text{occluded}(u, v)}$$

If masking a specific signature or number drastically drops $P(\text{Forged})$, that patch is verified as causal to the model's prediction.

---

## 3. Feature-Level Explainability (SHAP & LIME)

While deep CNN backbones excel at spatial feature extraction, forensic examiners rely on explicit physical signals (compression noise, edge gradients, toner density). We extract a deterministic 64-dimensional feature vector and explain it via game theory:

### 3.1 TreeSHAP (SHapley Additive exPlanations)
Based on cooperative game theory, Shapley values allocate fair payouts among features according to their marginal contributions across all possible feature subsets $S \subseteq F \setminus \{i\}$:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

- **TreeSHAP Efficiency:** Computes exact Shapley values in polynomial time $\mathcal{O}(TLD^2)$ for decision trees.
- **Forensic Utility:** Pinpoints whether high ELA skewness, Laplacian blur disparity, or LBP micro-texture inconsistency was the dominant driver of the fraud determination.

### 3.2 LIME (Local Interpretable Model-Agnostic Explanations)
Constructs a local surrogate linear model $g \in G$ by drawing perturbed samples $z'$ around instance $x$, weighted by exponential kernel distance $\pi_x(z)$:

$$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

Used to validate whether local linear approximations agree with global TreeSHAP rankings.

---

## 4. Contrastive & Counterfactual Explanations

Counterfactuals answer the prescriptive forensic question:
> *"What minimal alteration to this document would reverse the suspicion of forgery?"*

### 4.1 Feature-Level Counterfactual Search
Given input feature vector $x$ where $f(x) = 1$ (Forged), we solve:

$$x^* = \arg\min_{x'} d(x, x') \quad \text{s.t.} \quad P(f(x') = 1) \le \tau_{\text{authentic}}$$

The resulting vector provides actionable diagnostic recommendations (e.g., "Decreasing edge transition variance by 38% and harmonizing local compression residuals reduces forgery confidence below 40%").

### 4.2 Spatial Inpainting Counterfactual
Using the predicted localization mask $\hat{M}$, Telea fast-marching inpainting reconstructs the suspicious bounding box using authentic background textures, creating an explanatory visualization of what the document would look like in an unmanipulated state.

---

## 5. Case-Based Reasoning: Latent Exemplar Retrieval

For non-expert end-users, heatmap attributions can be abstract. Example-based XAI retrieves the most similar verified cases from the training corpus:
1. Extract the penultimate 2048-dimensional feature embedding $z = \phi(x)$.
2. Compute cosine similarity against indexed training documents:
   $$\text{Sim}(z_q, z_i) = \frac{z_q \cdot z_i}{\|z_q\|_2 \|z_i\|_2}$$
3. Return top-$k$ nearest verified authentic documents and top-$k$ nearest verified forged documents (ensuring master document self-retrieval is excluded).

---

## 6. Quantitative Evaluation of Explanation Quality

Rather than judging explanations by subjective aesthetic appeal, we establish four quantitative benchmarks:

### 6.1 Faithfulness (Deletion and Insertion AUC)
- **Deletion Test:** Iteratively mask the top $k\%$ most attributed pixels. If the attribution is faithful, model confidence must drop precipitously (yielding low Deletion AUC).
- **Insertion Test:** Start with a neutral blurred canvas and iteratively paste the top $k\%$ most attributed pixels. If the explanation captures essential forensic cues, confidence must rapidly recover (yielding high Insertion AUC).

### 6.2 Localization Agreement (IoU and Dice against Ground Truth)
We binarize the top 15% quantile of attribution heatmaps and compute IoU and Dice overlap against ground-truth tampering masks $M$:

$$\text{IoU}_{\text{XAI}} = \frac{|\text{Attr}_{\text{top}} \cap M|}{|\text{Attr}_{\text{top}} \cup M|}, \quad \text{Dice}_{\text{XAI}} = \frac{2 |\text{Attr}_{\text{top}} \cap M|}{|\text{Attr}_{\text{top}}| + |M|}$$

### 6.3 Perturbation Stability
Attributions must not fluctuate wildly under imperceptible scanner noise. We inject Gaussian noise ($\sigma = 0.02$) and compute Pearson correlation:

$$\rho = \frac{\text{Cov}(\text{Attr}(x), \text{Attr}(x + \epsilon))}{\sigma_{\text{Attr}(x)} \sigma_{\text{Attr}(x + \epsilon)}}$$

### 6.4 Attribution Sparsity (Gini Coefficient)
Measures whether the explanation concentrates on suspect regions rather than diffusing across irrelevant text. Higher Gini index reflects sharper focus.
