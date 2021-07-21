import torch
import torch.nn as nn
import torch.nn.functional as F


class Ner(nn.Module):
    def __init__(self, params):
        super(Ner, self).__init__()

        # maps each token to an embedding_dim vector
        self.embedding = nn.Embedding(
            params['vocab_size'], params['embedding_dim'])

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(params['embedding_dim'],
                            params['lstm_hidden_dim'], batch_first=True)

        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(params['lstm_hidden_dim'],
                            params['number_of_tags'])

    def forward(self, s):
        # apply the embedding layer that maps each token to its embedding
        # dim: batch_size x batch_max_len x embedding_dim
        s = self.embedding(s)

        # run the LSTM along the sentences of length batch_max_len
        # dim: batch_size x batch_max_len x lstm_hidden_dim
        s, _ = self.lstm(s)

        # reshape the Variable so that each row contains one token
        # dim: batch_size*batch_max_len x lstm_hidden_dim
        s = s.reshape(-1, s.shape[2])

        # apply the fully connected layer and obtain the output for each token
        s = self.fc(s)          # dim: batch_size*batch_max_len x num_tags

        # dim: batch_size*batch_max_len x num_tags
        return F.log_softmax(s, dim=1)
