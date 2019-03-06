import pandas as pd
import numpy as np
import os
import nltk
from nltk import word_tokenize
import nltk
import spacy
from spacy.lang.en import English
from nltk import Tree

os.chdir(os.path.dirname(__file__))

table = pd.read_table('./test_stage_1.tsv')
table.to_csv('test_stage_1.csv', index = False)

def word_tags(mention):
    """
    the type of the mention
    """
    tags  = nltk.pos_tag(mention)
    _, types = tags

    return types

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

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

class Embeding_features():

    def __init__(self):

        self.nlp = spacy.load('en_core_web_sm')

    def create(self, charoffset, text):

        doc = self.nlp(text)

        lens = [token.idx for token in doc] #The Charactor offset the token within the parent
        mention_offset = bs(lens, charoffset) - 1 # The target in which index of tokens
        mention = doc[mention_offset] #mention,

        dependency_parent = mention.head #The syntactic parent, or "governor", of this token.

        sent_lens = [len(sent) for sent in doc.sents] #the sentence length
        acc_lens = sent_lens
        pre_lens = 0
        for i in range(0, len(sent_lens)):
            pre_lens += acc_lens[i]
            acc_lens[i] = pre_lens

        sent_index = bs(acc_lens, mention_offset) #to Find out the charoffset in which sentence
        print(sent_index)
        mention_sent = list(doc.sents)[sent_index]

        first_word, last_word = mention_sent[0], mention_sent[-1]

        preceding2 = self.n_preceding_words(2, doc, mention_offset)
        following2 = self.n_following_words(2, doc, mention_offset)

        preceding5 = self.n_preceding_words(5, doc, mention_offset)
        following5 = self.n_following_words(5, doc, mention_offset)

        sent_tokens = [token for token in mention_sent]

        return mention, dependency_parent, first_word, last_word, preceding2, following2, preceding5, following5, sent_tokens


    def n_preceding_words(self, n, tokens, offset):

        start = offset-n-1
        start = max(0, start)
        end = offset

        return tokens[start:end]

    def n_following_words(self, n, tokens, offset):

        end = offset+n+1
        end = min(end, len(tokens))
        start = offset+1

        return tokens[start:end]

class Distance_features():

    def __init__(self):

        self.nlp = spacy.load('en_core_web_sm')
        self.buckets = [1, 2, 3, 4, 5, 8, 16, 32, 64]
        self.pos_buckets = [0, 1, 2, 3, 4, 5, 8, 16, 32]

    def create(self, char_offsetA, char_offsetB, text):

        doc = self.nlp(text)

        lens = [token.idx for token in doc]
        mention_offsetA = bs(lens, char_offsetA) - 1
        mention_offsetB = bs(lens, char_offsetB) - 1

        dist = mention_offsetA - mention_offsetB
        dist_oh = self.one_hot(self.buckets, dist)
        print(dist)
        print(dist_oh)

        sent_lens = [len(sent) for sent in doc.sents] #the sentence length
        acc_lens = sent_lens
        pre_lens = 0
        for i in range(0, len(sent_lens)):
            pre_lens += acc_lens[i]
            acc_lens[i] = pre_lens

        sentA_index = bs(acc_lens, mention_offsetA)
        sentB_index = bs(acc_lens, mention_offsetB)

        sentA = list(doc.sents)[sentA_index]
        sentB = list(doc.sents)[sentB_index]
        print(doc.sents)

        posA = mention_offsetA + 1
        if sentA_index > 0:
            posA = mention_offsetA - acc_lens[sentA_index-1] #The Distance from first word to mention
        posA_oh = self.one_hot(self.pos_buckets, posA)
        posA_end = len(sentA) - posA #The Distance from last word to mention In sentence

        posA_end_oh = self.one_hot(self.pos_buckets, posA_end)

        pos2 = mention_offsetB + 1
        if sentB_index > 0:
            posB = mention_offsetB - acc_lens[sentB_index-1]
        posB_oh = self.one_hot(self.pos_buckets, posB)
        posB_end = len(sentB) - posB #The Distance from last word to mention
        posB_end_oh = self.one_hot(self.pos_buckets, posB_end)

        sent_pos_ratioA = posA / len(sentA)
        sent_pos_ratioB = posB / len(sentB)

        return dist_oh, posA_oh, posB_oh, posA_end_oh, posB_end_oh

    def one_hot(self, lens, dist):

        low, high = 0, len(lens)

        while low < high:
            mid = low + int((high-low) / 2)
            if dist > lens[mid]:
                low = mid + 1
            elif dist < lens[mid]:
                high = mid
            else:
                idx = mid
                break

        idx = low
        if idx > len(lens) - 1:
            idx = len(lens) - 1
        one_hot = np.zeros(len(self.buckets))
        one_hot[idx] = 1

        return one_hot


Embeding_features().create(table['Pronoun-offset'][4], table['Text'][4])





#
