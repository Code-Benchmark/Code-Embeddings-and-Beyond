import Dataloaders.utils as utils
import pandas as pd
class DataLoader():
    def __init__(self, data_path):
        self.data_path = data_path
    def tbcnnDataloader(self):
        train_data = utils.load_tbcnn_data(self.data_path,'/example_trees.pkl','/example_embedding.pkl','train')
        test_data = utils.load_tbcnn_data(self.data_path, '/example_trees.pkl', '/example_embedding.pkl', 'test')
        return train_data, test_data

    def treecapsDataloader(self):
        train_data = utils.load_treecaps_data(self.data_path,'/example_trees.pkl','/nodemap.pkl','train')
        test_data = utils.load_treecaps_data(self.data_path, '/example_trees.pkl', '/nodemap.pkl', 'test')
        return train_data, test_data

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