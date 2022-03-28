import argparse
import TreeCaps.treecaps as treecaps
import Dataloaders.dataloaders as dataloaders
import torch

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='treecaps', help="The name of model(tbcnn|treecaps|code2vec|code2seq|astnn|ggnn|transformer|lstm|...)")
    parser.add_argument('--batch_size', type=int, default=1, help='train batch size, always 1')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dataset_directory', default="TreeCaps/data", help='the path of dataset')
    parser.add_argument('--model_path', default="TreeCaps/data/treecaps_model", help='path to save the model')
    parser.add_argument('--USE_GPU', default=True, type=bool, help='use gpu')
    parser.add_argument('--cuda', default="0", type=str, help='enables cuda')
    args = parser.parse_args()

    return args
if __name__ == '__main__':

    args = parse_args()
    dataloader = dataloaders.DataLoader(args.dataset_directory)
    train_data, test_data = getattr(dataloader, args.model_name+"Dataloader")()
    eval(args.model_name).train(args, train_data, test_data)