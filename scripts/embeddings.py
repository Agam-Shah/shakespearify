import torch
from transformers import BertTokenizer, BertModel


bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def create_sent_bert_embeddings(sentence):
    '''
    Function to create Sentence level BERT Embeddings(Mean Pooling)

:    Args:
        sentence: original and modern english cleaned sentence
    Returns:
        Embeddings vector of 768D-(Dimension)
    '''
    tokens = bert_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**tokens)

    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()


# if __name__ == "__main__":
#     pass
#     print(create_sent_bert_embeddings("Hello, How are you?"))