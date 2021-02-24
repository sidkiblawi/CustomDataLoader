import os  # loading files
import pandas as pd  # annotation
import spacy  # tokenization

import torch
from torch.nn.utils.rnn import pad_sequence  # for padding
from torch.utils.data import Dataset, DataLoader  # for loading data

spacy_eng = spacy.load('en')


class Vocabulary(object):
    def __init__(self, freq_threshold):
        self.stoi = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.stoi)

    @staticmethod
    def tokenize_str(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        index = len(self.stoi)

        for sentence in sentence_list:
            for token in self.tokenize_str(sentence):
                frequencies.setdefault(token, 0)
                frequencies[token] += 1

                if frequencies[token] >= self.freq_threshold:
                    self.stoi[token] = index
                    self.itos[index] = token
                    index += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize_str(text)

        return [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenized_text]


class CustomDataset(Dataset):
    def __init__(self, filename, freq_threshold=1):
        # load data
        self.df = pd.read_csv(filename)
        self.text = self.df['text']

        # Initialize and Build Vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.text.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]

        numericalized_text = [self.vocab.stoi['<BOS>']]
        numericalized_text += self.vocab.numericalize(text)
        numericalized_text.append(self.vocab.stoi['<EOS>'])

        return torch.tensor(numericalized_text)


class CustomCollate(object):
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):
        targets = pad_sequence(batch, batch_first=True, padding_value=self.pad_index)
        return targets
