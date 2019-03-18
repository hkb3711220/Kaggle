import pandas as pd
import numpy as np
import spacy
import nltk
from boltons.iterutils import windowed
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

os.chdir(os.path.dirname(__file__))
table = pd.read_table('./test_stage_1.tsv')

def bs(lens, target):

    low, high = 0, len(lens) -1

    while low < high:
        mid = low + int((high - low) / 2)

        if target > lens[mid]:
            low = mid+1
        elif target < lens[mid]:
            high = mid
        else:
            return mid+1

    return low

def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]

class extract_spans():

    def __init__(self, max_len=50, L=10):

        self.nlp = spacy.load('en_core_web_lg')
        self.L   = L
        self.max_len = max_len

    def create(self, charoffsetM, charoffsetA, charoffsetB, text):

        doc = self.nlp(text)

        charoffset = [charoffsetM, charoffsetA, charoffsetB]
        spans_list = []
        sent_pos = []

        for offset in charoffset:
            sent_features = self.extract(offset, doc)
            spans_list.append(sent_features[0])
            sent_pos.append(sent_features[1])

        return spans_list, sent_pos

    def extract(self, charoffset, doc):

        lens = [token.idx for token in doc] #The Charactor offset the token within the parent
        mention_offset = bs(lens, charoffset) - 1 # The target in which index of tokens
        mention = doc[mention_offset]

        #In this competiton, We already konwn that which mention We should pickup,
        #So the function of create_spans_index will no be used.

        sent_lens = [len(sent) for sent in doc.sents] #the sentence length
        acc_lens = sent_lens
        pre_lens = 0
        for i in range(0, len(sent_lens)):
            pre_lens += acc_lens[i]
            acc_lens[i] = pre_lens

        sent_index = bs(acc_lens, mention_offset)
        sent = list(doc.sents)[sent_index]

        pos = mention_offset+1
        if sent_index > 0:
            pos = mention_offset - acc_lens[sent_index-1]

        #idx_spans = self.create_idx_spans(doc)
        #idx_spans = list(set(idx_spans))
        #print(len(idx_spans))

        sent_features = []
        if len(sent) > self.max_len:

            if pos < self.max_len - 1:
                sent_features.append(sent[0:self.max_len].text)
            else:
                sent_features.append(sent[pos-self.max_len+2 : min(pos+2, len(sent))].text)
        else:
            sent_features.append(sent.text)
        sent_features.append(pos)
        #print(sent_features)

        return sent_features

    def create_idx_spans(self, text):

        """
        make a text into serverl spans depends on maximum span width L.
        In this competiton, We have already konwn that which mention We should pick up,
        So the function of create_idx_spans will not be used.

        the basic script is from https://github.com/shayneobrien/coreference-resolution/blob/master/src/utils.py

        """
        idx_spans, shift = [], 0
        while shift < len(text):
            candi_spans = flatten([windowed(range(shift, len(text)+shift), length) for length in range(1, self.L)])
            idx_spans.extend(candi_spans)
            shift += 1

        return idx_spans


def extract_sent_features(df, text_column, pronoun_offset_column, A_offset_column, B_offset_column):
    text_offset_list = df[[text_column, pronoun_offset_column, A_offset_column, B_offset_column]].values.tolist()
    extractor = extract_spans()
    pronoun_spans  = []
    A_offset_spans = []
    B_offset_spans = []
    pronoun_pos    = []
    A_offset_pos   = []
    B_offset_pos   = []

    for text_offset_index in range(len(text_offset_list)):
        text_offset = text_offset_list[text_offset_index]
        spans_list, sent_pos = extractor.create(text_offset[1], text_offset[2], text_offset[3], text_offset[0])
        pronoun_spans.append(spans_list[0])
        A_offset_spans.append(spans_list[1])
        B_offset_spans.append(spans_list[2])
        pronoun_pos.append(sent_pos[0])
        A_offset_pos.append(sent_pos[1])
        B_offset_pos.append(sent_pos[2])

    return pronoun_spans, A_offset_spans, B_offset_spans, pronoun_pos, A_offset_pos, B_offset_pos

pronoun_spans, A_offset_spans, B_offset_spans, pronoun_pos, A_offset_pos, B_offset_pos = extract_sent_features(table, 'Text', 'Pronoun-offset', 'A-offset', 'B-offset')



all_spans = pronoun_spans +  A_offset_spans + B_offset_spans
tokenizer = Tokenizer(num_words=80000)
tokenizer.fit_on_texts(list(all_spans))

tr_X = tokenizer.texts_to_sequences(pronoun_spans)
tr_X = pad_sequences(tr_X, maxlen=50)
print(tr_X.shape)
