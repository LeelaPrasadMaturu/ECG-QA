import torch
from PIL import Image
from torchvision import transforms
from configs import config
from model.fusion_network import ClinicalFusionNetwork

class ClinicalECGPredictor:
    def __init__(self, model_path):
        self.model = ClinicalFusionNetwork()
        self.model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        self.model.to(config.DEVICE)
        self.model.eval()
        
        self.tokenizer = ClinicalTextProcessor().tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(config.IMG_MEAN, config.IMG_STD)
        ])
    
    def preprocess_ecg_image(self, image_path):
        img = Image.open(image_path).convert('L')
        return self.transform(img).unsqueeze(0).to(config.DEVICE)
    
    def preprocess_question(self, text):
        return self.tokenizer(
            text,
            max_length=config.MAX_SEQ_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(config.DEVICE)
    
    def analyze_ecg(self, image_path, question):
        ecg_image = self.preprocess_ecg_image(image_path)
        inputs = self.preprocess_question(question)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = self.model(ecg_image, inputs.input_ids, inputs.attention_mask)
            probs = torch.softmax(logits, dim=1)
            
        return {
            'normal': probs[0][0].item(),
            'abnormal': probs[0][1].item(),
            'critical': probs[0][2].item()
        }