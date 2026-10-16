\# Cross-Modal Image-Caption Retrieval System Using Dual-Encoder Model



A robust multimodal deep learning system that bridges visual perception and natural language processing by mapping images and text captions into a shared 512-dimensional vector space using asymmetric dual-encoders, enabling efficient bidirectional semantic search (Image-to-Text and Text-to-Image).



\---



\## 👥 Authors

\* \*\*Abhishek Mundanat\*\* (PES1UG24AM012)\[cite: 3]

\* \*\*Ankitha\*\* (PES1UG24AM044)\[cite: 3]



\---



\## 🚀 Project Overview

Traditional search engines operate exclusively within a single modality, making it impossible to directly query or index raw visual content using text, or vice versa. This project solves that challenge by implementing a \*\*Dual-Encoder Architecture\*\* trained via contrastive learning. 



Key capabilities of the system:

1\. \*\*Bidirectional Retrieval\*\*: Supports both \*\*Image-to-Text (I2T)\*\* search (querying an image to retrieve matching captions) and \*\*Text-to-Image (T2I)\*\* search (querying text descriptions to retrieve matching images)\[cite: 3].

2\. \*\*Shared Semantic Embedding Space\*\*: Projects disparate modalities into an aligned 512-dimensional space where similarity is efficiently computed via cosine similarity (dot product of L2-normalized vectors)\[cite: 3].

3\. \*\*Measurable Benchmarking\*\*: Evaluated quantitatively using \*\*Recall@K (R@1, R@5, R@10)\*\* and \*\*Mean Reciprocal Rank (MRR)\*\*\[cite: 3].



\---



\## 📂 Repository Structure (`src/`)

\* \*\*`dataset.py`\*\*: Handles loading the Flickr8k dataset, image resizing/standardization, text lowercasing, tokenization, and sequence padding\[cite: 3].

\* \*\*`model.py`\*\*: Implements the `DualEncoderModel` class combining a pre-trained \*\*ResNet-50\*\* vision backbone and a \*\*DistilBERT\*\* text backbone with linear projection heads and L2 normalization\[cite: 3].

\* \*\*`train.py`\*\*: Manages the training loop, \*\*Triplet/Contrastive Loss\*\* computation with in-batch negatives, and optimization via the \*\*AdamW optimizer\*\*\[cite: 3].

\* \*\*`evaluate.py`\*\*: Computes global retrieval metrics (Recall@1, R@5, R@10, MRR) across both I2T and T2I directions and includes a dynamic random-index qualitative sample demo\[cite: 3].



\---



\## 📊 Dataset Description

The system is trained and evaluated on the benchmark \*\*Flickr8k Dataset\*\*:

\* \*\*Images\*\*: 8,000 RGB images capturing everyday scenes, activities, and objects\[cite: 3].

\* \*\*Captions\*\*: 40,000 total text annotations (exactly 5 diverse human-written captions per image)\[cite: 3].



\---



\## 🛠️ System Architecture \& Methodology



\### 1. Modality-Specific Encoders

\* \*\*Vision Encoder\*\*: Pre-trained Convolutional Neural Network (\*\*ResNet-50\*\*) (with classification head removed) extracts spatial visual features, followed by adaptive pooling and a linear projection to output an L2-normalized 512-dimensional vector ($v$)\[cite: 3].

\* \*\*Text Encoder\*\*: Pre-trained Transformer model (\*\*DistilBERT\*\*) processes tokenized text sequences, performing mean pooling over token hidden states followed by a linear projection to output an L2-normalized 512-dimensional vector ($t$)\[cite: 3].



\### 2. Training Objective

Optimized using \*\*Triplet Ranking Loss (Contrastive Learning)\*\*:

$$\\mathcal{L} = \\max(0, d(A, P) - d(A, N) + \\alpha)$$

\* Pulls matching image-caption pairs closer together while pushing mismatches apart by margin $\\alpha$, leveraging in-batch negatives for computational efficiency\[cite: 3].



\---



\## 📈 Evaluation Results \& Performance



Model performance benchmarked on the test split:



| Retrieval Task | Recall@1 | Recall@5 | Recall@10 | MRR |

| :--- | :---: | :---: | :---: | :---: |

| \*\*Image-to-Text (I2T)\*\* | 20.00% | 94.00% | 100.00% | 0.4499\[cite: 3] |

| \*\*Text-to-Image (T2I)\*\* | 16.00% | 100.00% | 100.00% | 0.4203\[cite: 3] |



\* \*\*Strong Top-K Recall\*\*: The system successfully places correct matches within the top 5 results for 94% (I2T) and 100% (T2I) of queries\[cite: 3].

\* \*\*Fine-Grained Ranking Challenge\*\*: While broad semantic neighborhoods are captured exceptionally well, Recall@1 reflects minor score variations due to close semantic overlap among alternative human-written captions for the same image\[cite: 3].



\---



\## ⚙️ Installation \& Setup



1\. \*\*Clone the Repository\*\*:

&#x20;  ```bash

&#x20;  git clone \[https://github.com/Abhishek8016/Image-Caption-Retrieval.git](https://github.com/Abhishek8016/Image-Caption-Retrieval.git)

&#x20;  cd Image-Caption-Retrieval

