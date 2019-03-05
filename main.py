import argparse
import torch

parser = argparse.ArgumentParser("TSA w/ LSTM")

parser.add_argument('--data',type=str)
parser.add_argument('--model',type=str,default='lstm-max')
parser.add_argument('--window',type=int,help='the window size using for perdicion',default=50)
parser.add_argument('--predict_size',type=int,help='the period to perdict',default=1)
parser.add_argument('--batch',type=int,default=16)
parser.add_argument('--epoch',type=int,default=10)

gpu = torch.cuda.is_available()

if __name__ == '__main__':
    pass

