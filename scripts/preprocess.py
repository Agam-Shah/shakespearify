import re
import spacy
from datasets import load_dataset
import pandas as pd
import os

nlp = spacy.load("en_core_web_sm")

def preprocess_dataset(file):
    '''
    Function to load the dataset using 'datasets' module

    Args:
        file_path: csv file location
    Returns:
        returns dataset
    '''
    dataset_dict = load_dataset("csv", data_files=file)
    dataset = dataset_dict['train']
    dataset = dataset.map(preprocess_row, num_proc = 4)
    return dataset


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

    return row

if __name__ == "__main__":

    dir_path = "C:/Users/ASUS/OneDrive/Desktop/shakespearify"
    file_path = os.path.join(dir_path,"data", "final_v3.csv")
    dataset = preprocess_dataset(file_path)
    print(dataset[0:2])
