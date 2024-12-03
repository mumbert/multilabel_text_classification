from datasets import Dataset, load_dataset, DatasetDict
import random
import utils
import pandas as pd

class MyDataset():

    def __init__(self, json_file, csv_file, num_examples = -1, verbose = False):

        self.json_file = json_file
        self.csv_file = csv_file
        self.verbose = verbose

        self.json_data = utils.load_json_file(self.json_file)
        json_data = Dataset.from_list(self.json_data)
        csv_data = pd.read_csv(self.csv_file)
        csv_data["target"] -= 1 # from 1-4 to 0-3
        merged_data = pd.DataFrame(json_data).merge(csv_data, on='id')
        if num_examples > -1:
            merged_data = merged_data.head(num_examples)

        self.num_examples = merged_data.shape[0]
        self.dataset = Dataset.from_pandas(merged_data)

        if self.verbose:
            print(f"* Loaded dataset with {self.num_examples} samples")
            print(f"\t- There are {len(list(merged_data['target'].unique()))} different targets")
            print(list(merged_data['target'].unique()))

    def __call__(self, seed = 0, train_size = 0.5, valid_size = 0.25, test_size = 0.25):

        self.seed = seed
        self.test_size = test_size

        assert train_size+valid_size+test_size == 1.0

        # Split the dataset into train (80%) and test (20%)
        train_test_split = self.dataset.train_test_split(train_size=train_size, seed=self.seed)

        # Further split the test set into validation (10%) and test (10%)
        test_valid_split = train_test_split['test'].train_test_split(test_size=test_size/(1-train_size), seed=self.seed)

        # Combine splits into a DatasetDict
        self.dataset_splits = DatasetDict({
            'train': train_test_split['train'],
            'validation': test_valid_split['train'],
            'test': test_valid_split['test']
        })

        if self.verbose:
            print(f"Dataset structure")
            print(self.dataset_splits)

