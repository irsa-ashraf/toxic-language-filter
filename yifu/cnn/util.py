# This is a Python file for helper functions for `cnn.ipynb`

from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np
import torch

BATCH_SIZE = 16
MAX_SENT_LENGTH = 200
EMBED_DIM = 300

def process_input(batch):
    '''
    Tokenize a batch of texts:
        - convert sentence to lower case
        - tokenize word with package `word_tokenize`
        - add paddings to MAX_SENT_LENGTH
        - convert tokenized sentence into GloVe embeddings
        - return a numpy array of input vectors 
    '''

    # set up containers
    y = torch.zeros(BATCH_SIZE)
    x = torch.zeros(BATCH_SIZE, MAX_SENT_LENGTH, EMBED_DIM)
    
    for i, (sent, label) in enumerate(batch):

        sent = sent.lower()
        tokenized_sent = word_tokenize(sent)

        # perform padding or truncate
        if len(tokenized_sent) < MAX_SENT_LENGTH:
            tokenized_sent += ['<pad>'] * (MAX_SENT_LENGTH - len(tokenized_sent))
        else:
            tokenized_sent = tokenized_sent[:MAX_SENT_LENGTH]

        vecs = glove.get_vecs_by_tokens(tokenized_sent)

        x[i] = vecs
        y[i] = label

    return x, y 