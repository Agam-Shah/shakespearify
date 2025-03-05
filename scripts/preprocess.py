import re
import spacy
from datasets import load_dataset
import pandas as pd
import os
from embeddings import create_sent_bert_embeddings

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    '''
    Function to remove extra spaces between words and make all the words in lowercase!
    Keeps the Punctuation!

    Args:
        text: each datarow of csv file
    Returns:
        returns each cleaned datarow.
    '''
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_text(text):
    '''
    Function to tokenize the dataset using spacy

    Args:
        text: each cleaned datarow from csv file
    Returns:
        tokenize list of each cleaned datarow
    '''
    tokenize = []
    doc = nlp(text)

    for token in doc:
        tokenize.append(token.text)

    return tokenize

def preprocess_row(row):
    '''
    Function to preprocess each datarow using already specified functions above

    Args:
        row: each raw datarow from the csv file
    Returns:
        returns cleaned as well as tokenized rows
    '''
    row['original'] = clean_text(row['original'])
    row['modern'] = clean_text(row['modern'])
    row['original_tokens'] = tokenize_text(row['original'])
    row['modern_tokens'] = tokenize_text(row['modern'])
    row['original_embedding'] = create_sent_bert_embeddings(row['original'])
    row['modern_embedding'] = create_sent_bert_embeddings(row['modern'])

    return row

def preprocess_and_save_dataset(file):
    '''
    Function to load the dataset using 'datasets' module and save the embeddings

    Args:
        file_path: csv file location
    Returns:
        returns processed dataset
    '''

    save_dir = os.path.join(os.getcwd(),"data", "processed")
    os.makedirs(save_dir, exist_ok=True)
    dataset_dict = load_dataset("csv", data_files=file)
    dataset = dataset_dict['train']
    dataset = dataset.map(preprocess_row, num_proc = 4)
    dataset.save_to_disk(save_dir)
    print("Dataset processed and saved")
    return dataset



if __name__ == "__main__":

    dir_path = os.getcwd()
    file_path = os.path.join(dir_path,"data", "final_v3.csv")
    dataset = preprocess_and_save_dataset(file_path)
