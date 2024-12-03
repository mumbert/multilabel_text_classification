import preprocess
from dataset import MyDataset
import glove
import model
import torch
import pytorch_lightning as pl
import utils
import pandas as pd
import os

def get_dataset(config):

    # LOAD DATASET
    data = MyDataset(json_file = config["dataset"]["json_file"],
                     csv_file = config["dataset"]["csv_file"],
                     num_examples = config["dataset"]["num_examples"],
                     verbose = config["verbose"])
    data(seed = config["dataset"]["seed"], 
         train_size = config["dataset"]["train_size"], 
         valid_size = config["dataset"]["valid_size"], 
         test_size = config["dataset"]["test_size"])
    
    # PROCESS DATASET
    dataset = preprocess.preprocess_dataset(data.dataset_splits)

    return dataset

def get_vocab(dataset, config):

    vocab = preprocess.build_vocab(dataset = dataset, 
                                MAX_TOKENS=config["preprocess"]["max_tokens"], 
                                verbose = config["verbose"])
    
    return vocab

def get_dataset_split(dataset, config):

    train_dataloader, valid_dataloader, test_dataloader, test_dataloader_with_ids = preprocess.create_dataloaders(dataset, 
                                                                                        BATCH_SIZE = config["model"]["batch_size"])
    
    return train_dataloader, valid_dataloader, test_dataloader, test_dataloader_with_ids

def get_model_trained(vocab, train_dataloader, valid_dataloader, test_dataloader, config):

    # PREPARE GLOVE PRETRAINED EMBEDDINGS
    word_vectors_glove = glove.load_glove_embeddings(config["model"]["glove_file_path"], config["model"]["word2vec_file_path"])

    # TRAIN MODEL using PyTorch Lightning Trainer
    model_with_glove_unfreeze = model.SentimentModel(vocab_size=len(vocab), 
                                                     embed_dim=config["model"]["embed_dim"], 
                                                     output_dim=config["model"]["output_dim"], 
                                                     pretrained_embeddings=torch.tensor(word_vectors_glove.vectors), 
                                                     freeze_embeddings=False)

    # TRAIN+VALIDATE, TEST
    trainer = pl.Trainer(max_epochs=config["model"]["num_epochs"])
    trainer.fit(model_with_glove_unfreeze, train_dataloader, valid_dataloader)
    trainer.test(model_with_glove_unfreeze, test_dataloader)

    # Write test results per class
    # softmax = torch.nn.Softmax(dim=1)
    # res = []
    # ids = []
    # for text, id in test_dataloader_with_ids:
    #     prob = softmax(model_with_glove_unfreeze.forward(text))
    #     prob = prob.cpu().detach().tolist()
    #     res.extend(prob)
    #     ids.extend(id)
    # df = pd.DataFrame(ids, columns=["id"])
    # df[["0", "1", "2", "3"]] = prob
    # df.to_csv(config["model"]["predictions_file"], index=False)

    return model_with_glove_unfreeze

def get_predictions(model_with_glove_unfreeze, test_dataloader_with_ids, config):

    print(f"* Saving test predictions to csv file: {config['model']['predictions_file']}")

    softmax = torch.nn.Softmax(dim=1)
    res = []
    ids = []
    for text, id in test_dataloader_with_ids:
        prob = softmax(model_with_glove_unfreeze.forward(text))
        prob = prob.cpu().detach().tolist()
        res.extend(prob)
        ids.extend(id)
    df = pd.DataFrame(ids, columns=["id"])
    df[["0", "1", "2", "3"]] = res
    os.makedirs(os.path.dirname(config["model"]["predictions_file"]), exist_ok = True)
    df.to_csv(config["model"]["predictions_file"], index=False)

def get_model_saved(model_with_glove_unfreeze, vocab, config):

    utils.save_checkpoint(checkpoint_file = config["model"]['checkpoint_file'], 
                          model = model_with_glove_unfreeze,
                          vocab = vocab,
                          EMBED_DIM = config["model"]["embed_dim"],
                          OUTPUT_DIM = config["model"]["output_dim"])