import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import numpy as np

os.chdir(os.path.dirname(__file__))

class DataLoader(object):

    def __init__(self, path, name):
        self.path = path
        self.data_path = os.path.join(path, '{}.txt'.format(name))

        self.data = self._load_data(self.data_path)
        if name == 'train':
            self.word_training = list(set(self.data))
            self.char_training = []
            #Convert $w_t$ into a series of characters: $w_t = [c_1, c_2, ..., c_N]$
            for word in self.word_training:
                for char in word:
                    if char not in self.char_training:
                        self.char_training.append(char)

            self.word_index, self.char_index = self._create_dict(self.word_training, self.char_training)

        self.inputs, self.outputs = self.prepare_input_output(self.data)

    def _load_data(self, path):

        f = open(path, 'r')
        text = f.read().split()

        return text

    def _create_dict(self, words, chars):

        word_index = {}
        char_index = {}

        for i, word in enumerate(words):
            word_index[word] = i

        for i, char in enumerate(chars):
            char_index[char] = i+1
        char_index[''] = 0

        return word_index, char_index

    def prepare_input_output(self, data):

        output_words = []
        input_words = []
        input_chars = []
        sequence_length = []


        for i in range(len(data)-1):
            output_words.append(data[i+1])
            input_words.append(data[i])

        for n in range(len(input_words)):
            input_chars.append([c for c in input_words[n]])

        return input_chars, output_words

def generator(inputs,
             outputs,
             batch_size,
             max_len,
             max_word_len,
             word_index,
             char_index):

    I = 0
    while True:
        x = np.zeros(shape=(batch_size, max_len, max_word_len))
        y = np.zeros(shape=(batch_size, max_len, len(word_index)))

        for sample in range(batch_size):
            for time in range(max_len):
                if I >= len(inputs):
                    I = 0

                for n, c in enumerate(inputs[I]):
                    x[sample, time, n] =  char_index[c]

                y[sample, time, word_index[outputs[I]]] = 1.0

                I +=1

        yield x, y
