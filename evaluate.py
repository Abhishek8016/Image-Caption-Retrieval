import os
import warnings
import random  # <--- 1. Import random module

# Suppress Hugging Face progress bars, warnings, and unnecessary terminal noise
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, Subset
from dataset import FlickrDataset, get_transforms
from model import DualEncoderModel


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    captions_file = "data/captions.txt" if os.path.exists("data/captions.txt") else "data/captions.csv"
    img_dir = "data/images"
    checkpoint_path = "checkpoints/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        return

    full_dataset = FlickrDataset(captions_file=captions_file, img_dir=img_dir, transform=get_transforms())
    subset_indices = list(range(min(50, len(full_dataset))))
    dataset = Subset(full_dataset, subset_indices)
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    model = DualEncoderModel(embed_size=512).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    all_img_embs = []
    all_txt_embs = []
    candidate_captions = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                captions = batch['caption']
            else:
                images = batch[0].to(device)
                captions = batch[1]
                
            img_emb, txt_emb = model(images, captions, device)
            all_img_embs.append(img_emb)
            all_txt_embs.append(txt_emb)
            candidate_captions.extend(captions)
            
    img_embeddings = torch.cat(all_img_embs, dim=0)
    txt_embeddings = torch.cat(all_txt_embs, dim=0)
    
    sim_matrix = torch.matmul(img_embeddings, txt_embeddings.T)
    n = sim_matrix.size(0)
    targets = torch.arange(n, device=device)
    
    # Recall@K and MRR
    i2t_ranks = torch.argsort(sim_matrix, dim=1, descending=True)
    i2t_r1 = (i2t_ranks[:, :1] == targets.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    i2t_r5 = (i2t_ranks[:, :5] == targets.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    i2t_r10 = (i2t_ranks[:, :10] == targets.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    
    # Compute Mean Reciprocal Rank (MRR)
    i2t_correct_pos = (i2t_ranks == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
    i2t_mrr = (1.0 / i2t_correct_pos.float()).mean().item()
    
    t2i_ranks = torch.argsort(sim_matrix.T, dim=1, descending=True)
    t2i_r1 = (t2i_ranks[:, :1] == targets.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    t2i_r5 = (t2i_ranks[:, :5] == targets.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    t2i_r10 = (t2i_ranks[:, :10] == targets.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    
    t2i_correct_pos = (t2i_ranks == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
    t2i_mrr = (1.0 / t2i_correct_pos.float()).mean().item()

    # --- 2. Pick a random query index within the evaluation subset range ---
    query_idx = random.randint(0, n - 1)
    sample_scores, sample_indices = torch.topk(sim_matrix[query_idx], k=3)

    # FINAL DISPLAY OUTPUT
    print("\n======================================================================")
    print("                CROSS-MODAL RETRIEVAL EVALUATION RESULTS                ")
    print("======================================================================")
    print("Task                   | Recall@1 | Recall@5 | Recall@10 |    MRR    ")
    print("-----------------------+----------+----------+-----------+------------")
    print(f"Image-to-Text (I2T)    |  {i2t_r1:5.2f}%  |  {i2t_r5:5.2f}%  |    {i2t_r10:5.2f}%  |  {i2t_mrr:.4f}")
    print(f"Text-to-Image (T2I)    |  {t2i_r1:5.2f}%  |  {t2i_r5:5.2f}%  |    {t2i_r10:5.2f}%  |  {t2i_mrr:.4f}")
    print("======================================================================\n")

    print("======================================================================")
    print("SAMPLE RETRIEVAL DEMO")
    print("======================================================================")
    print(f"Query Image Index : {query_idx}")  # <--- 3. Print the randomized index
    print("\nTop Retrieved Captions:")
    for rank, (score, idx) in enumerate(zip(sample_scores, sample_indices), start=1):
        print(f"  Rank {rank} ({score.item():.4f}): \"{candidate_captions[idx]}\"")
    print("======================================================================\n")


if __name__ == "__main__":
    evaluate()