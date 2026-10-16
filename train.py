import os
import warnings

# Suppress Hugging Face warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, Subset
from dataset import FlickrDataset, get_transforms
from model import DualEncoderModel
from loss import TripletRankingLoss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running training on device: {device}")
    
    captions_file = "data/captions.txt" if os.path.exists("data/captions.txt") else "data/captions.csv"
    img_dir = "data/images"
    
    print("Loading full dataset...")
    full_dataset = FlickrDataset(captions_file=captions_file, img_dir=img_dir, transform=get_transforms())
    
    # ------------------------------------------------------------------
    # REDUCE DATASET: Slice to exactly 100 image-caption samples
    # ------------------------------------------------------------------
    subset_indices = list(range(112))
    dataset = Subset(full_dataset, subset_indices)
    
    # batch_size=16 with drop_last=False gives exactly 4 batches per epoch
    dataloader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=True, 
        drop_last=False, 
        num_workers=0
    )
    
    model = DualEncoderModel(embed_size=512).to(device)
    criterion = TripletRankingLoss(margin=0.2)
    
    # Train linear projection layers only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
    
    epochs = 10
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"\n--- Starting Super-Fast Demo Training (112 Images, {epochs} Epochs) ---")
    print(f"Total Batches per Epoch: {len(dataloader)}\n")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(dataloader, start=1):
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                captions = batch['caption']
            else:
                images = batch[0].to(device)
                captions = batch[1]
                
            optimizer.zero_grad()
            
            img_emb, txt_emb = model(images, captions, device)
            loss = criterion(img_emb, txt_emb)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Print every step for real-time progress
            print(f"Epoch [{epoch}/{epochs}] | Step [{step}/{len(dataloader)}] | Batch Loss: {loss.item():.4f}", flush=True)
                
        avg_loss = total_loss / len(dataloader)
        print(f"===> Epoch [{epoch}/{epochs}] Completed | Average Loss: {avg_loss:.4f}\n", flush=True)
        
    torch.save(model.state_dict(), "checkpoints/best_model.pth")
    print("Fast demo training complete! Model weights saved to 'checkpoints/best_model.pth'.")

if __name__ == "__main__":
    train()