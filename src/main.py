import argparse
import utils
import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", help="Config file", type=str)

def training_pipeline(config):

    # LOAD DATASET
    dataset = pipeline.get_dataset(config)
 
    # GET VOCABULARY
    vocab = pipeline.get_vocab(dataset, config)

    # SPLIT DATASETS
    train_dataloader, valid_dataloader, test_dataloader, test_dataloader_with_ids = pipeline.get_dataset_split(dataset, config)
    
    # TRAIN, VALIDATE, TEST
    model_with_glove_unfreeze = pipeline.get_model_trained(vocab, train_dataloader, valid_dataloader, test_dataloader, config)

    # OUTPUT PREDICTIONS
    pipeline.get_predictions(model_with_glove_unfreeze, test_dataloader_with_ids, config)

    # SAVE CHECKPOINT
    pipeline.get_model_saved(model_with_glove_unfreeze, vocab, config)

if __name__ == "__main__":

    # LOAD CONFIG
    args = parser.parse_args()
    config = utils.load_yaml_file(args.config_file, show = True)

    # LAUNCH TRAINING PIPELINE
    training_pipeline(config)
