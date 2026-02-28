import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoTokenizer, AutoModel
from model import MultiModalModel
import torch.nn as nn
import tqdm
import random

def convert_digits_to_random_text(labels):
    digit_to_text = {
        '0': ['0', 'Zero', 'zero', 'Null', '00'],
        '1': ['1', 'One', 'one', 'Won', 'Onew'],
        '2': ['2', 'Two', 'two', 'Too', 'ToO'],
        '3': ['3', 'Three', 'tree', 'Thre', 'tHREE'],
        '4': ['4', 'Four', 'four', 'For', 'fOUR'],
        '5': ['5', 'Five', 'quinque', 'cinco', 'Paanch'],
        '6': ['6', 'Six', 'six', 'SIX', 'siX'],
        '7': ['7', 'Seven', 'seven', 'SeVeN', 'SEVEN'],
        '8': ['8', 'Eight', 'eight', 'EIGHT', 'eiGht'],
        '9': ['9', 'Nine', 'nine', 'NINE', 'NiNe']
    }

    return [random.choice(digit_to_text[str(label.item())]) for label in labels]

def batched_tokenizer(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Training pipeline
class Trainer:
    def __init__(self, learning_rate=0.001, batch_size=256, num_epochs=5):
        # Device setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Model
        self.model = MultiModalModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CosineEmbeddingLoss()

        # Tokenizer and BERT model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)

        # Data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def fit(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for images, labels in tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                # Convert labels to random text and tokenize
                random_texts = convert_digits_to_random_text(labels)
                text_embeddings = self.tokenizer(random_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                text_features = self.bert_model(**text_embeddings).last_hidden_state[:, 0, :]

                # Forward pass
                image_embeddings, text_embeddings = self.model(images, text_features)
                target = torch.ones(images.size(0)).to(self.device)  # Similarity target
                loss = self.criterion(image_embeddings, text_embeddings, target)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(self.train_loader):.4f}")

if __name__ == "__main__":
    trainer = Trainer()

    trainer.fit()



