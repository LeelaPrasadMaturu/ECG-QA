import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from configs import config
from model.fusion_network import ClinicalFusionNetwork

class ClinicalTrainer:
    def __init__(self):
        self.model = ClinicalFusionNetwork().to(config.DEVICE)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY)
        self.scaler = GradScaler(enabled=config.AMP_ENABLED)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            images = batch['image'].to(config.DEVICE)
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)
            
            with autocast(enabled=config.AMP_ENABLED):
                outputs = self.model(images, input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # ... similar to training ...
                
        return total_loss / len(val_loader), correct / total

    def save_checkpoint(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }, path)

def main():
    from data_processing.data_loader import load_clinical_data
    
    trainer = ClinicalTrainer()
    train_loader, val_loader = load_clinical_data()
    
    best_acc = 0.0
    patience = 0
    
    for epoch in range(config.NUM_EPOCHS):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.validate(val_loader)
        
        # Early stopping and checkpointing
        if val_acc > best_acc:
            trainer.save_checkpoint(config.CHECKPOINT_PATH)
            best_acc = val_acc
            patience = 0
        else:
            patience += 1
            
        if patience >= config.EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

if __name__ == "__main__":
    main()