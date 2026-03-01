import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalModel(nn.Module):
    def __init__(self, image_embedding_dim=128, text_embedding_dim=128):
        super(MultiModalModel, self).__init__()
        
        # Image Encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 7 x 7
            nn.Flatten(),                          # Flatten to a vector
            nn.Linear(64 * 7 * 7, image_embedding_dim),  # Map to image embedding
            nn.ReLU()
            nn.ReLU()
        )
        
        # Text Encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(768, 256),  # Input: BERT's hidden size
            nn.ReLU(),
            nn.Linear(256, text_embedding_dim),  # Map to text embedding
            nn.ReLU()
        )
    
    def forward(self, images, text_embeddings):
        image_embedding = self.image_encoder(images)
        text_embedding = self.text_encoder(text_embeddings)
        

        return image_embedding, text_embedding
