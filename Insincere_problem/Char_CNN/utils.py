import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import numpy as np

path = r'C:\Users\user1\Desktop\Kaggle\Insincere_problem\Input'
max_len = 35

class DataLoader(object):

    def __init__(self, path, max_len):
        self.path = path
        self.max_len = max_len
        self.data_path = os.path.join(path, 'small_vocab_en')

        self.train_data = self._load_data(self.data_path)

        text = []
        for sequence in self.train_data:
            [text.append(word) for word in sequence]
        self.word_training = list(set(text))
        self.char_training = []
        #Convert $w_t$ into a series of characters: $w_t = [c_1, c_2, ..., c_N]$
        for word in self.word_training:
            for char in word:
                if char not in self.char_training:
                    self.char_training.append(char)

        self.word_index, self.char_index = self._create_dict(self.word_training, self.char_training)
        self.input_chars, self.output_words = self.prepare_input_output(self.train_data)
        self.inputs, self.outputs = self.create_input_output(self.input_chars, self.output_words)

    def _load_data(self, path):

        f = open(path, 'r')
        f = [sequence.replace('\n', ' ') for sequence in f]
        text = [re.sub("[^\w]", " ", sequence).split() for sequence in f]

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


        for sequence in data:
            for i in range(len(sequence)-1):
                output_words.append(sequence[i+1])
                input_words.append(sequence[i])

        for n in range(len(input_words)):
            input_chars.append([c for c in input_words[n]])

        return input_chars, output_words

    def create_input_output(self, input_chars, output_words):
        """
        #In the paper we see that training is done in groups of seq_len=30 timesteps
        """

        inputs = []
        outputs = []

        seq_len = self.max_len

        for n in range(0, len(input_chars)-seq_len, seq_len):
            inputs.append(input_chars[n:n+seq_len])
            outputs.append(output_words[n:n+seq_len])

        return inputs, outputs

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
                for chars in inputs[I]:
                    for n, c in enumerate(chars):
                        x[sample, time, n] =  char_index[c]

                for word in outputs[I]:

                    y[sample, time, word_index[word]] = 1.0

                I +=1

        yield x, y
