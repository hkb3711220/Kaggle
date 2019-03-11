import pandas as pd
import numpy as np
import os
import nltk
import spacy
from nltk import Tree

os.chdir(os.path.dirname(__file__))
table = pd.read_table('./test_stage_1.tsv')

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

        self.nlp = spacy.load('en_core_web_lg')

    def create(self, charoffset, text):

        doc = self.nlp(text)

        lens = [token.idx for token in doc] #The Charactor offset the token within the parent
        mention_offset = bs(lens, charoffset) - 1 # The target in which index of tokens
        mention = doc[mention_offset] #mention

        dependency_parent = mention.head #The syntactic parent, or "governor", of this token.
        children = mention.children

        sent_lens = [len(sent) for sent in doc.sents] #the sentence length
        acc_lens = sent_lens
        pre_lens = 0
        for i in range(0, len(sent_lens)):
            pre_lens += acc_lens[i]
            acc_lens[i] = pre_lens

        sent_index = bs(acc_lens, mention_offset) #to Find out the charoffset in which sentence
        current_sent = list(doc.sents)[sent_index]

        preceding3 = self.n_preceding_words(3, doc, mention_offset)
        following3 = self.n_following_words(3, doc, mention_offset)

        proceeding = []
        for i in range(sent_index-2, sent_index):
            if i < 0 :
                continue
            else:
                proceeding.append(list(doc.sents)[i])

        if sent_index+1 < len(list(doc.sents)):
            succeeding = list(doc.sents)[sent_index+1]

        return mention, dependency_parent, preceding3, following3, proceeding, current_sent, succeeding

    def n_preceding_words(self, n, tokens, offset):

        start = offset - n
        precedings = [None] * max(0, 0 - start)
        start = max(0, start)
        precedings += tokens[start: offset]

        return precedings

    def n_following_words(self, n, tokens, offset):

        end = offset + n
        followings = [None] * max(0, end - len(tokens))
        end = min(end, len(tokens))
        followings += tokens[offset: end]

        return followings

mention, dependency_parent, childern, preceding3, following3, proceeding, current_sent, succeeding = Embeding_features().create(table["Pronoun-offset"][0], table["Text"][0])
print(mention, dependency_parent, preceding3, following3, proceeding, current_sent, succeeding)


class Distance_features():

    def __init__(self):

        self.nlp = spacy.load('en_core_web_lg')
        self.buckets = [1, 2, 3, 4, 5, 8, 16, 32, 64]
        self.pos_buckets = [0, 1, 2, 3, 4, 5, 8, 16, 32]

    def create(self, char_offsetA, char_offsetB, text):

        doc = self.nlp(text)

        lens = [token.idx for token in doc]
        mention_offsetA = bs(lens, char_offsetA) - 1
        mention_offsetB = bs(lens, char_offsetB) - 1

        dist = mention_offsetA - mention_offsetB
        dist_oh = self.one_hot(self.buckets, dist)

        sent_lens = [len(sent) for sent in doc.sents] #the sentence length
        acc_lens = sent_lens
        pre_lens = 0
        for i in range(0, len(sent_lens)):
            pre_lens += acc_lens[i]
            acc_lens[i] = pre_lens

        sentA_index = bs(acc_lens, mention_offsetA)
        sentB_index = bs(acc_lens, mention_offsetB)

        sent_dist = sentA_index - sentB_index

        sentA = list(doc.sents)[sentA_index]
        sentB = list(doc.sents)[sentB_index]
        #print(doc.sents)

        posA = mention_offsetA + 1
        if sentA_index > 0:
            posA = mention_offsetA - acc_lens[sentA_index-1] #The Distance from first word to mention
        posA_oh = self.one_hot(self.pos_buckets, posA)
        posA_end = len(sentA) - posA #The Distance from last word to mention In sentence

        posA_end_oh = self.one_hot(self.pos_buckets, posA_end)

        posB = mention_offsetB + 1
        if sentB_index > 0:
            posB = mention_offsetB - acc_lens[sentB_index-1]
        posB_oh = self.one_hot(self.pos_buckets, posB)
        posB_end = len(sentB) - posB #The Distance from last word to mention
        posB_end_oh = self.one_hot(self.pos_buckets, posB_end)

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


def extract_embed_features(df, text_column, offset_column, num_embed_features = 11, embed_dim = 300):
    text_offset_list = df[[text_column, offset_column]].values.tolist()
    num_features     = num_embed_features
    extractor        = Embeding_features()

    embed_feature_matrix = np.zeros(shape=(len(text_offset_list), num_features, embed_dim))
    for text_offset_index in range(len(text_offset_list)):
        text_offset = text_offset_list[text_offset_index]
        mention, parent, first_word, last_word, precedings2, followings2, precedings5, followings5, sent_tokens = extractor.create(text_offset[1], text_offset[0])

        feature_index = 0
        embed_feature_matrix[text_offset_index, feature_index, :] = mention.vector
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = parent.vector
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = first_word.vector
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = last_word.vector
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index:feature_index+2, :] = np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in precedings2])
        feature_index += len(precedings2)
        embed_feature_matrix[text_offset_index, feature_index:feature_index+2, :] = np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in followings2])
        feature_index += len(followings2)
        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in precedings5]), axis=0)
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in followings5]), axis=0)
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(np.asarray([token.vector for token in sent_tokens]), axis=0) if len(sent_tokens) > 0 else np.zeros(embed_dim)
        feature_index += 1

    return embed_feature_matrix


def extract_dist_features(df, text_column, pronoun_offset_column, name_offset_column, num_features=45):
    text_offset_list = df[[text_column, pronoun_offset_column, name_offset_column]].values.tolist()
    extractor = Distance_features()

    pos_feature_matrix = np.zeros(shape=(len(text_offset_list), num_features))
    for text_offset_index in range(len(text_offset_list)):
        text_offset = text_offset_list[text_offset_index]
        dist_oh, sent_pos_oh1, sent_pos_oh2, sent_pos_inv_oh1, sent_pos_inv_oh2 = extractor.create(text_offset[1], text_offset[2], text_offset[0])

        feature_index = 0
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(dist_oh)] = np.asarray(dist_oh)
        feature_index += len(dist_oh)
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_oh1)] = np.asarray(sent_pos_oh1)
        feature_index += len(sent_pos_oh1)
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_oh2)] = np.asarray(sent_pos_oh2)
        feature_index += len(sent_pos_oh2)
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_inv_oh1)] = np.asarray(sent_pos_inv_oh1)
        feature_index += len(sent_pos_inv_oh1)
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_inv_oh2)] = np.asarray(sent_pos_inv_oh2)
        feature_index += len(sent_pos_inv_oh2)

    return pos_feature_matrix
