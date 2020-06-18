import torch
import torch.nn as nn
import pickle

class SentimentPredictor():
    def __init__(self):
        with open('config.pickle', 'rb') as handle:
            self.config = pickle.load(handle)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.TwitterSentimentPredictor(self.config['input_dim'], 
                                               self.config['embedding_dim'],
                                               self.config['hidden_dim'],
                                               self.config['output_dim'],
                                               self.config['n_layers'],
                                               self.config['pad_idx'])
        self.model.load_state_dict(torch.load('twitter_sentiment_state_dict.pt', map_location=self.device))
        self.model.to(self.device)
    
    def clean(self, text):
        USER = '@[\w_]+'
        LINK = 'https?:\/\/\S+'
        HASHTAG = '#\S+'
        NUMBER = '\d+'
        PUNCTUATIONS = '[\.?!,;:\-\[\]\{\}\(\)\'\"/]'

        user_sub = re.sub(USER, ' <user> ', text)
        link_sub = re.sub(LINK, ' <url> ', user_sub)
        hashtag_sub = re.sub(HASHTAG, ' <hashtag> ', link_sub)
        number_sub = re.sub(NUMBER, ' <number> ', hashtag_sub)
        clean_text = re.sub(PUNCTUATIONS, ' ', number_sub)

        return clean_text.lower()

    def predict(self, text):
        clean_text = self.clean(text)
        tokenized = [word for word in clean_text.split(' ') if word != '']

        indexed = [self.config['stoi'][w] for w in tokenized]
        length = [len(indexed)]
        self.model.eval()
        with torch.no_grad():
            tensor = torch.LongTensor(indexed).to(device)
            tensor = tensor.unsqueeze(1)
            length_tensor = torch.LongTensor(length)

            prediction = torch.sigmoid(self.model(tensor, length_tensor))

        return prediction.item()

    
    class TwitterSentimentPredictor(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, pad_idx):
            super(SentimentPredictor.TwitterSentimentPredictor, self).__init__()

            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

            # Define LSTM Layers
            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout=0.8, bidirectional=True)

            # Final dense
            self.dropout = nn.Dropout(0.8)
            self.dense1 = nn.Linear(hidden_dim * 2, output_dim)
        

        def forward(self, input, lengths):

            embed = self.embedding(input)

            # Pack sequence
            packed_embed = nn.utils.rnn.pack_padded_sequence(embed, lengths)
            packed_output, (hidden, cell) = self.lstm(packed_embed)

            # Take the hidden state of the last layer for output
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
            out = self.dense1(hidden)

            return out
