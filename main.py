import argparse
import torch
from data_process import DataProcessor
from models.lstm_max import LstmMax
from tqdm import tqdm
import torch.nn as nn
import torch
import logging

parser = argparse.ArgumentParser("TSA w/ LSTM")
parser.add_argument('--data',type=str)
parser.add_argument('--model',type=str,default='lstm-max')
parser.add_argument('--window',type=int,help='the window size using for perdicion',default=50)
parser.add_argument('--predict_size',type=int,help='the period to perdict',default=1)
parser.add_argument('--batch',type=int,default=16)
parser.add_argument('--epoch',type=int,default=10)
parser.add_argument('--lstmhidden',type=int,default=1024)
parser.add_argument('--dp',type=float,default=0.1)
parser.add_argument('--bi',type=bool,default=True)
parser.add_argument('--log_path',type=str,default='log.log')

gpu = torch.cuda.is_available()
param, _ = parser.parse_known_args()
min_loss = float('inf')

logging.basicConfig(filename=param.log_path, format='%(asctime)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':
    data = DataProcessor(param.data, window_size=param.window, output_size=param.predict_size,
                         split_ratio=0.2, shuffle=True, random_state=1109)
    data.process()
    X_train, y_train, X_test, y_test = data.split()

    # X.shape = [-1,window,number of stocks]
    # y.shape = [-1, predict period, number of stocks]

    model = LstmMax(X_train.shape[-1], param.lstmhidden, param.dp, param.bi)
    if gpu:
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    def valiate(test=False):
        global min_loss
        with torch.no_grad():
            loss = 0
            for i in tqdm(range(0, X_test.shape[0], param.batch * 2), desc='Validate', ncols=0):
                x = torch.tensor(X_test[i:i+param.batch * 2]).float()
                y = torch.tensor(y_test[i:i+param.batch * 2]).float()
                if gpu:
                    x = x.cuda()
                    y = y.cuda()
                out = model(x)
                loss += loss_fn(out,y.view(out.shape))
            if float(loss) < min_loss:
                min_loss = float(loss)
                torch.save(model.state_dict(),'save.model')
                logging.warning('save model at {}'.format(round(float(loss),4)))


    def train():
        for ep in range(param.epoch):
            for i in tqdm(range(0, X_train.shape[0], param.batch), desc='EP {}:'.format(ep), ncols=0):
                x = torch.tensor(X_train[i:i+param.batch]).float()
                y = torch.tensor(y_train[i:i+param.batch]).float()
                if gpu:
                    x = x.cuda()
                    y = y.cuda()
                out = model(x)
                optim.zero_grad()
                loss = loss_fn(out,y.view(out.shape))
                loss.backward()
                optim.step()

            valiate()
    try:
        train()
    except KeyboardInterrupt:
        valiate()




