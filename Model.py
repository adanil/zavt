import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

train_on_gpu = False
alphabet_size = 256

class LSTMNet(nn.Module):
    def __init__(self, alphabet_size, n_hidden=256, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.alphabet_size = alphabet_size

        self.lstm = nn.LSTM(alphabet_size, n_hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden, alphabet_size)

    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)

        out = r_output.contiguous().view(-1, self.n_hidden)

        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden


def one_hot_encode(arr,n_labels):
    one_hot = np.zeros((arr.size,n_labels),dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape,n_labels))
    return one_hot

def load_model(path):
    with open(path, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))

    loaded = LSTMNet(checkpoint['alphabet_size'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    loaded.load_state_dict(checkpoint['state_dict'])
    return loaded


def predict(net, symbol, h=None):
    x = np.array(symbol)
    x = one_hot_encode(x, alphabet_size)
    inputs = torch.from_numpy(x)

    if train_on_gpu:
        inputs = inputs.cuda()

    h = tuple([each.data for each in h])

    inputs = inputs.reshape((1, 1, 256))
    out, h = net(inputs, h)

    return F.softmax(out, dim=1).data, h


def process(net, size, bytes):
    if (train_on_gpu):
        net.cuda()
    else:
        net.cpu()

    net.eval()  # eval mode

    # First off, run through the prime characters
    h = net.init_hidden(1)
    for i in range(size):
        b = bytes[i]
        next = 0
        if i < size - 1:
            next = bytes[i + 1]
        prob, h = predict(net, b, h)
        np.set_printoptions(suppress=True)
        prob = prob.flatten().cpu().numpy()
        print('Elem prob: ', prob[next])
