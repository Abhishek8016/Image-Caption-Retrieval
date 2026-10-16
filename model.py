import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertTokenizer, DistilBertModel

class DualEncoderModel(nn.Module):
    def __init__(self, embed_size=512):
        super(DualEncoderModel, self).__init__()
        
        # 1. ResNet-50 Image Backbone (Frozen for fast CPU execution)
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
            
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()  # Remove default ImageNet classifier
        self.image_encoder = resnet
        self.img_projection = nn.Linear(in_features, embed_size)
        
        # 2. DistilBERT Text Backbone (Frozen for fast CPU execution)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        self.txt_projection = nn.Linear(self.text_encoder.config.hidden_size, embed_size)

    def forward(self, images, captions, device):
        # Image Feature Extraction
        img_features = self.image_encoder(images)
        img_embeddings = self.img_projection(img_features)
        
        # Safe Text Processing & Tokenization
        if isinstance(captions, (list, tuple)):
            tokens = self.tokenizer(
                list(captions), 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(device)
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
        elif isinstance(captions, dict):
            input_ids = captions['input_ids'].to(device)
            attention_mask = captions.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        else:
            input_ids = captions.to(device)
            attention_mask = None
            
        # Text Feature Extraction
        if attention_mask is not None:
            txt_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            txt_outputs = self.text_encoder(input_ids=input_ids)
            
        txt_features = txt_outputs.last_hidden_state[:, 0, :]  # Extract CLS token representation
        txt_embeddings = self.txt_projection(txt_features)
        
        # L2 Normalization onto shared embedding space
        img_embeddings = nn.functional.normalize(img_embeddings, p=2, dim=1)
        txt_embeddings = nn.functional.normalize(txt_embeddings, p=2, dim=1)
        
        return img_embeddings, txt_embeddings