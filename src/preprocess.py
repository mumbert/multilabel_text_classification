from torchtext.vocab import build_vocab_from_iterator

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

# Initialize stopwords for English and the WordNet lemmatizer
STOP_WORDS = set(stopwords.words('english'))  # Use a set for faster lookup
LEMMATIZER = WordNetLemmatizer()
SPECIAL_TOKENS = ["<unk>"]  # Special tokens for handling unknown words

vocab = None

def preprocess_example(example):
    """
    Preprocess a single example for use with the Hugging Face datasets map method.

    Steps:
    1. Tokenize the text using NLTK's word tokenizer.
    2. Convert words to lowercase.
    3. Remove stopwords.
    4. Lemmatize words.

    Args:
    - example (dict): A dictionary containing a "text" field with the sentence to process.

    Returns:
    - dict: A dictionary with the text replaced by preprocessed tokens.
    """
    # Tokenize the text
    words = word_tokenize(str(example))

    # Preprocess words: lowercase, remove stopwords, and lemmatize
    preprocessed_tokens = [
        LEMMATIZER.lemmatize(word.lower())
        for word in words
        if word.lower() not in STOP_WORDS
    ]

    # Return the updated example with tokens
    return {"tokens": preprocessed_tokens}

def preprocess_dataset(dataset):

    # Apply preprocessing to the dataset
    dataset = dataset.map(
        preprocess_example,
        num_proc=4  # Use multiprocessing for efficiency
    )

    return dataset


def yield_tokens(data_iter, token_field="tokens"):
    """
    Generator to yield tokens from a dataset for vocabulary building.

    Args:
    - data_iter (iterable): An iterable dataset (e.g., a training split).
    - token_field (str): The key in the dataset containing the tokenized data.

    Yields:
    - list of str: Tokens from each example in the dataset.
    """
    for example in data_iter:
        yield example[token_field]

def build_vocab(dataset, MAX_TOKENS = 25000, verbose = False):
    # Build the vocabulary from the training dataset
    global vocab
    vocab = build_vocab_from_iterator(
        yield_tokens(dataset["train"], token_field="tokens"),
        specials=SPECIAL_TOKENS,
        max_tokens=MAX_TOKENS
    )

    # Set default index for unknown tokens
    vocab.set_default_index(vocab["<unk>"])

    if verbose:
        # Print vocabulary statistics
        print(f"Vocabulary size: {len(vocab)}")
        print(f"First 10 tokens in vocabulary: {list(vocab.get_itos()[:50])}")

    return vocab

def collate_batch(batch):
    """
    Collate function to prepare batches for the DataLoader.

    Args:
    - batch (list of dict): A batch of examples, where each example is a dictionary
                            containing "tokens" (list of str) and "label" (int or float).

    Returns:
    - padded_texts (torch.Tensor): A tensor of shape (batch_size, max_seq_length) with
                                   token indices padded to the same length.
    - labels (torch.Tensor): A tensor of shape (batch_size,) with labels as float values.
    """
    # Define processing pipelines for text and labels
    text_pipeline = lambda tokens: vocab(tokens)  # Convert tokens to indices using the vocabulary
    label_pipeline = lambda label: torch.tensor(label, dtype=torch.long)  # Convert labels to float tensors

    # Apply pipelines to create tensors for texts and labels
    text_tensors = [torch.tensor(text_pipeline(item["tokens"])) for item in batch]
    # label_tensors = [label_pipeline(item["target"]) for item in batch]
    label_tensors = [torch.nn.functional.one_hot(label_pipeline(item["target"]), num_classes=4) for item in batch]

    # Pad text sequences to the same length and convert labels to a tensor
    padded_texts = pad_sequence(text_tensors, batch_first=True)
    # labels = torch.tensor(label_tensors)
    labels = torch.stack(label_tensors, dim=0)

    return padded_texts, labels

def collate_batch_with_ids(batch):

    # Define processing pipelines for text and labels
    text_pipeline = lambda tokens: vocab(tokens)  # Convert tokens to indices using the vocabulary

    # Apply pipelines to create tensors for texts and labels
    text_tensors = [torch.tensor(text_pipeline(item["tokens"])) for item in batch]
    ids = [item["id"] for item in batch]

    # Pad text sequences to the same length and convert labels to a tensor
    padded_texts = pad_sequence(text_tensors, batch_first=True)

    return padded_texts, ids

def create_dataloaders(dataset, BATCH_SIZE = 64):

    # vocab = VOCAB

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle the training dataset for better generalization
        collate_fn=collate_batch  # Use the custom collate function
    )

    valid_dataloader = DataLoader(
        dataset["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,  # No need to shuffle the validation dataset
        collate_fn=collate_batch
    )

    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,  # No need to shuffle the test dataset
        collate_fn=collate_batch
    )

    test_dataloader_with_ids = DataLoader(
        dataset["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,  # No need to shuffle the test dataset
        collate_fn=collate_batch_with_ids
    )

    return train_dataloader, valid_dataloader, test_dataloader, test_dataloader_with_ids