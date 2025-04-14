import torch
from pathlib import Path

BASE_DATA_DIR = Path("../clinical_ecg_data")

class ClinicalConfig:
   TEMPLATE_STRUCTURE = {
        "input_splits": ["train", "valid", "test"],
        "image_filename": lambda ecg_id: f"ECG{ecg_id}_clinical_512.png",
        "path_mapping": {
            "raw_templates": BASE_DATA_DIR/"raw"/"original_templates",
            "processed_images": BASE_DATA_DIR/"processed"/"EcgImages",
            "updated_templates": BASE_DATA_DIR/"processed"/"updated_templates"
        }
    }
    
    # Image parameters
    IMG_SIZE = 512
    IMG_MEAN = [0.485]
    IMG_STD = [0.229]
    
    # Text processing
    CLINICAL_BERT_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    MAX_SEQ_LENGTH = 128
    
    # Model architecture
    IMG_EMBED_SIZE = 512
    TEXT_EMBED_SIZE = 768
    NUM_CLASSES = 3  # Normal/Abnormal/Critical
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    EARLY_STOP_PATIENCE = 5
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    AMP_ENABLED = True

config = ClinicalConfig()