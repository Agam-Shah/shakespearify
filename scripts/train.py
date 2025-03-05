import os
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from model import BERT2Transformer


# from data_loader import train_dataloader

def load_processed_data():
    """
    Load the processed data from the disk

    Returns:
        returns the processed data
    """
    save_dir = os.path.join(os.getcwd(), "data", "processed")
    dataset = load_from_disk(save_dir)
    print(f"**********Loaded {len(dataset)} preprocessed samples**********\n")
    # print(dataset[0])
    return dataset


# Custom PyTorch Dataset to ensure uniform batching
class ShakespeareDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Convert embeddings to PyTorch tensors
        modern_embedding = torch.tensor(sample["modern_embedding"], dtype=torch.float32)
        shakespeare_embedding = torch.tensor(sample["original_embedding"], dtype=torch.float32)

        return {"modern_embedding": modern_embedding, "shakespeare_embedding": shakespeare_embedding}


if __name__ == '__main__':

    # Load the processed data
    dataset = load_processed_data()


    # Define batch size
    BATCH_SIZE = 16

    # Wrap dataset in PyTorch's Dataset class
    train_dataset = ShakespeareDataset(dataset)

    # Use DataLoader for batching & shuffling
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define Device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Model
    model = BERT2Transformer().to(device)

    # Define Loss & Optimizer
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    # Load Checkpoint (If Resuming)
    checkpoint_path = os.path.join(os.getcwd(), "models", "shakespeare_model.pth")
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded existing model checkpoint!")
    except FileNotFoundError:
        print("No checkpoint found. Starting fresh training.")

    # Training Loop
    NUM_EPOCHS = 10  # Adjust based on resources

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            modern_embedding = batch["modern_embedding"].to(device)
            shakespeare_embedding = batch["shakespeare_embedding"].to(device)

            # Define Target Tensor for Cosine Loss (1 means they should be similar)
            target = torch.ones(modern_embedding.size(0)).to(device)

            # Forward Pass
            optimizer.zero_grad()
            outputs = model(modern_embedding, shakespeare_embedding)
            outputs = outputs.squeeze(1)

            # Compute Loss
            loss = criterion(outputs, shakespeare_embedding, target)

            # Backpropagation
            loss.backward()

            # Update Weights
            optimizer.step()

            total_loss += loss.item()

            # Print loss every 10 steps
            if step % 10 == 0:
                print(f"Step [{step}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {total_loss / len(train_dataloader)}")

    # Get the directory to save the model
    model_dir = os.path.join(os.getcwd(), "models")

    # Create dir if it doesn't exists
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "shakespeare_model.pth")
    print(f"Saving model to: {save_path}")

    # Save the model
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully!")
