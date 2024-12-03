
# Multi-label Text Classification

This project aims to train a multi-label classifier with some sample data. The dataset used consists of a random sample of the arXiv dataset made publicly available on Kaggle, consisting of various submitted scholarly article metadata in a machine-readable format.

## Requirements

Before running the code, please:
- create the conda environment by executing: `bash create_env.sh`
- download the glove embeddings by executing `bash download_globe_embeddings.sh`

To run the code:
-  activate the conda environment `conda activate multilabel_text_classification` and run `python src/main.py --config_file config/test.yaml`
-  you can also test the code through the notebook `multilabel_text_classification.ipynb`

## System description

The system has been designed to follow multiple steps as implemented in function `main.py/training_pipeline()`

These steps are:
- `pipeline.py::get_dataset()`: to load the dataset
- `pipeline.py::get_vocab(dataset, config)`: to get the corresponding vocabulary which will actually be considered
- `pipeline.py::train_dataloader, valid_dataloader, test_dataloader = pipeline.get_dataset_split(dataset, vocab, config)`: to split the dataset into train, validation and test subsets
- `pipeline.py::pipeline.get_model_trained()`: to initialize the model architecture with glove pretrained embeddings and then finetune it with the train subset. It also tests the model with the test subset and prints the results
- `pipeline.py::pipeline.get_model_saved()`: to save the model checkpoint

## Configuration file

Some details on the multiple fields in the configuration file. I am taking the default one as an example in `config/test.yaml`

```
dataset:                                                                # dataset related configuration values
  json_file: data/input/sample_data.json                                # input data as json file
  csv_file: data/input/sample_targets.csv                               # input labels as csv file
  train_size: 0.5                                                       # size of the training split
  valid_size: 0.25                                                      # size of the validation split
  test_size: 0.25                                                       # size of the test split
  seed: 0                                                               # internal seed value for dataset spliting
  num_examples: -1                                                      # amount of files to be loaded in total (-1 == all files)
preprocess:                                                             # preprocessing related configuration values
  max_tokens: 25000                                                     # Limit the maximum size of the vocabulary
model:                                                                  # model training related configuration values
  batch_size: 64                                                        # Define batch size
  glove_file_path: "data/glove/glove.6B/glove.6B.300d.txt"              # input glove pretrained embeddings
  word2vec_file_path: "data/glove/glove.6B/glove.6B.300d.word2vec.txt"  # output glove word2vec
  embed_dim: 300                                                        # embeddings dimensions
  output_dim: 4                                                         # number of predicted classes
  num_epochs: 10                                                        # number of training epochs
  checkpoint_file: "data/checkpoint/state_dict.pt"                      # checkpoint file path
  predictions_file: "data/predictions/results.csv"                      # predictions file path
verbose: true                                                           # verbose flag
```

## Dataset

Description of Data Features:
- id: The ArXiv ID
- submitter: Who submitted the paper
- authors: Authors of the paper
- title: Title of the paper
- comments: Additional info, such as number of pages and figures
- journal-ref: Information about the journal the paper was published in
- doi: [https://www.doi.org](Digital Object Identifier)
- abstract: The abstract of the paper
- categories: Categories / tags in the ArXiv system (Further details here: https://arxiv.org/category_taxonomy)
- versions: A version history
- update_date: Date of article update
- authors_parsed: Parsed authors metadata

Further details about arXiv can be found here: https://info.arxiv.org/about/index.html

## Results

The results obtained are on the test set, so these are the IDs reported in the CSV file that can be found at:

```
data/predictions/results.csv
```

A results example with an execution using the default configuration values is shown in the following table:

```
───────────────────────────────────────────────────
       Test metric             DataLoader 0
───────────────────────────────────────────────────                    
      test_accuracy         0.8895131349563599
        test_loss           0.7353142499923706
───────────────────────────────────────────────────
```

## Future work

These are some thoughts on what could be added having production in mind:
- pipeline components
- code design and maintainability
- model design
- evaluation metrics

Pipeline components:

- `data cleaning`: remove some data since not everything might be necessary, and thus shortening the input raw text. this process can also help to locate corrupt data for instance.
- `configuration`: extend the configuration file to manage hyperparameters and use `ray` to find the best possible architecture parameters and enhance flexibility.


Code design and maintainability:
- `logging`: provide logging information of the processes via logging module and redirect these messages to an output log file. this is also useful to manage execution time and data drift detection.
- `debugging`: perform a step by step analysis of the different pipeline components in order to ensure no errors are being made.
- `poetry for python environment and others`: create the environment using poetry instead of conda in order to ensure reproducibility and ease package management. also, the same poetry configuration file can be used to ensure the code uses descriptive variable names and consistent coding standards (e.g., PEP 8, black)
- `comments when needed`: Add comments and docstrings to explain logic and parameters.
- `unit testing`: to ensure code reliability, maintainability and facilitate refactoring. these would be created on a separate `tests` folder and `pytest` module can be used for that.
- `fastAPI`: Package the model in a deployable format for easy integration with production systems.
- `dockerization`: to ensure reproducibility and to make the system ready for production.

Model design:
- `baseline model`: build a simple baseline model that would be used as a reference

Evaluation metrics:
- `Choose relevant metrics`: like F1-score, precision, recall, and ROC-AUC for multi-label performance.
- `explainability`: explore if tools like SHAP or LIME help explain the results