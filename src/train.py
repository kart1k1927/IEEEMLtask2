import yaml
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Reproducibility
torch.manual_seed(config["seed"])

# Load model from Hugging Face
model = SentenceTransformer(config["model_name"])

# Dummy training data (replace later)
train_examples = [
    InputExample(texts=["Machine learning is fun", "I love studying ML"]),
    InputExample(texts=["Deep learning uses neural networks", "Neural nets power deep learning"]),
]

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=config["batch_size"]
)

train_loss = losses.MultipleNegativesRankingLoss(model)

# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=config["epochs"],
    warmup_steps=10
)

# Save model
model.save("models/fine_tuned")
