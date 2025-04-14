import torch.nn as nn
from .image_encoder import ClinicalImageEncoder
from .text_processor import ClinicalTextProcessor
from configs import config

class ClinicalFusionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ClinicalImageEncoder()
        self.text_processor = ClinicalTextProcessor()
        
        self.fusion = nn.Sequential(
            nn.Linear(config.IMG_EMBED_SIZE + config.TEXT_EMBED_SIZE, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, config.NUM_CLASSES))
        
    def forward(self, images, input_ids, attention_mask):
        img_emb = self.image_encoder(images)
        text_emb = self.text_processor(input_ids, attention_mask)
        combined = torch.cat([img_emb, text_emb], dim=1)
        return self.fusion(combined)