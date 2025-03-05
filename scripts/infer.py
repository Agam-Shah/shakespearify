import torch
import os
import torch.nn.functional as F
from embeddings import bert_tokenizer, bert_model
from model import BERT2Transformer
from datasets import load_from_disk

# Load processed Shakespearean data
processed_data_path = os.path.join(os.getcwd(), "data", "processed")
dataset = load_from_disk(processed_data_path)

# Load trained model
model_path = os.path.join(os.getcwd(), "models", "shakespeare_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BERT2Transformer()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


def generate_shakespearean(text):
    """Converts modern English text to Shakespearean style using trained model."""

    # Tokenize & create embeddings
    tokens = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embedding = bert_model(**tokens).last_hidden_state.mean(dim=1)  # (1, 768)

    embedding = embedding.unsqueeze(1).to(device)  # Reshape to (1, 1, 768)

    # Pass through model
    with torch.no_grad():
        output_embedding = model.transformer_decoder(embedding, embedding)  # (1, 1, 768)

    output_embedding = output_embedding.squeeze(1)  # Back to (1, 768)

    # Find closest match in Shakespearean dataset
    best_match = None
    best_score = -1

    for sample in dataset:
        shakespeare_embedding = torch.tensor(sample["modern_embedding"]).to(device)
        score = F.cosine_similarity(output_embedding, shakespeare_embedding, dim=-1).item()

        if score > best_score:
            best_score = score
            best_match = sample["modern"]

    return best_match


if __name__ == "__main__":
    input_text = input("Enter modern English text: ")
    print("\nShakespearean Translation:", generate_shakespearean(input_text))
