import Dataloaders.utils as utils
import pandas as pd
class DataLoader():
    def __init__(self, data_path):
        self.data_path = data_path
    def treecapsDataloader(self):
        train_data = pd.read_csv(self.data_path + '/train_example.csv')
        dev_data = pd.read_csv(self.data_path + '/dev_example.csv')
        test_data = pd.read_csv(self.data_path + '/test_example.csv')
        return train_data, dev_data, test_data

    def tbcnnDataloader(self):
    	pass

    def astnnDataloader(self):
    	pass

    def ggnnDataloader(self):
    	pass

    def code2vecDataloader(self):
    	pass

    def code2seqDataloader(self):
    	pass

    def tptransDataloader(self):
    	pass