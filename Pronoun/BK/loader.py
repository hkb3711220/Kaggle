import pandas as pd
import numpy as np
import os
import spacy
import tensorflow_hub as hub
import tensorflow as tf

os.chdir(os.path.dirname(__file__))
table = pd.read_table('./test_stage_1.tsv')
nlp   = spacy.load('en_core_web_lg')
elmo = hub.Module("https://tfhub.dev/google/elmo/2")

def bs(lens, target):
    low, high = 0, len(lens) - 1

    while low < high:
        mid = low + int((high - low) / 2)

        if target > lens[mid]:
            low = mid + 1
        elif target < lens[mid]:
            high = mid
        else:
            return mid + 1

    return low

class Mention_Features():

    def __init__(self):

        self.nlp = nlp

    def create(self, charoffset, text):

        doc = self.nlp(text)

        lens = [token.idx for token in doc]  # The Charactor offset the token within the parent
        mention_offset = bs(lens, charoffset) - 1  # The target in which index of tokens
        mention = doc[mention_offset]  # mention

        dependency_parent = mention.head  # The syntactic parent, or "governor", of this token.
        nbor = mention.nbor()  # The following word of nbor

        sent_lens = [len(sent) for sent in doc.sents]  # the sentence length
        acc_lens = sent_lens
        pre_lens = 0
        for i in range(0, len(sent_lens)):
            pre_lens += acc_lens[i]
            acc_lens[i] = pre_lens
        sent_index = bs(acc_lens, mention_offset)  # to Find out the charoffset which sentence
        current_sent = list(doc.sents)[sent_index]

        preceding3 = self.n_preceding_words(3, doc, mention_offset)
        following3 = self.n_following_words(3, doc, mention_offset)

        proceed_sents = []  # 3 proceeding sentence
        for i in range(sent_index - 3, sent_index):
            if i < 0: continue
            proceeding = [token for token in list(doc.sents)[i]]
            proceed_sents.extend(proceeding)

        if sent_index + 1 < len(list(doc.sents)):  # 1 succeeding sentence
            succeeding = list(doc.sents)[sent_index + 1]
            succeed_sent = [token for token in succeeding]
        else:
            succeed_sent = []

        return mention, dependency_parent, nbor, preceding3, following3, proceed_sents, current_sent, succeed_sent

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

class Distance_Features():

    def __init__(self):

        self.nlp = nlp

    def create(self, char_offsetA, char_offsetB, text):

        doc = self.nlp(text)

        lens = [token.idx for token in doc]
        mention_offsetA = bs(lens, char_offsetA) - 1
        mention_offsetB = bs(lens, char_offsetB) - 1
        mention_dist = mention_offsetA - mention_offsetB

        sent_lens = [len(sent) for sent in doc.sents] #the sentence length
        acc_lens = sent_lens
        pre_lens = 0
        for i in range(0, len(sent_lens)):
            pre_lens += acc_lens[i]
            acc_lens[i] = pre_lens

        sentA_index = bs(acc_lens, mention_offsetA)
        sentB_index = bs(acc_lens, mention_offsetB)

        sent_dist = sentA_index - sentB_index

        return [mention_dist, sent_dist]



def sentence_embed(tokens):

    if len(tokens) > 0:
        _tokens = [token.text for token in tokens]
        _tokens = np.asarray(_tokens)
        _tokens = np.expand_dims(_tokens, axis=0)
        sentence_embed = elmo(inputs={'tokens': _tokens, 'sequence_len': [_tokens.shape[1]]},
                              signature='tokens',as_dict=True)["elmo"]
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sentence_embeddings = session.run(sentence_embed)
            return np.mean(sentence_embeddings, axis=1)
    else:
        return np.zeros((1024,))

def extract_embedding_features(df, text_column, offset_column,  embed_dim=300):
    text_offset_list = df[[text_column, offset_column]].values.tolist()
    extractor = Mention_Features()

    feature_map1 = np.zeros(shape=(len(text_offset_list), 3, embed_dim+512))
    feature_map2 = np.zeros(shape=(len(text_offset_list), 6, embed_dim+512))
    feature_map3 = np.zeros(shape=(len(text_offset_list), 3, embed_dim+1024))

    for text_offset_index in range(len(text_offset_list)):
        text_offset = text_offset_list[text_offset_index]
        mention, dependency_parent, nbor, preceding3, following3, proceed_sents, current_sent, succeed_sent = extractor.create( text_offset[1], text_offset[0])
        first3 = []
        # Feature Map1
        feature_map1[text_offset_index, 0, :300] = dependency_parent.vector
        feature_map1[text_offset_index, 1, :300] = mention.vector
        feature_map1[text_offset_index, 2, :300] = nbor.vector
        first3.extend([dependency_parent.text, mention.text, nbor.text])
        first3_embed = elmo(first3, as_dict=True)['word_emb']
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            first3_embeddings   = session.run(first3_embed)
        feature_map1[text_offset_index, :, 300:812] = np.squeeze(first3_embeddings, axis=1)

        # Feature Map2
        feature_map2[text_offset_index, 0:3, :300] = np.asarray(
            [token.vector if token is not None else np.zeros((embed_dim,)) for token in preceding3])
        feature_map2[text_offset_index, 3:6, :300] = np.asarray(
            [token.vector if token is not None else np.zeros((embed_dim,)) for token in following3])
        p3_f3 = preceding3 + following3
        for i, token in enumerate(p3_f3): p3_f3[i] = token.text
        p3_f3_embed = elmo(p3_f3, as_dict=True)['word_emb']
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            p3_f3_embeddings   = session.run(p3_f3_embed)
        feature_map2[text_offset_index, :, 300:812] = np.squeeze(p3_f3_embeddings, axis=1)

        #Feature Map3
        feature_map3[text_offset_index, 0, :300] = np.mean(np.asarray([token.vector for token in proceed_sents]),
                                                        axis=0) if len(proceed_sents) > 0 else np.zeros(embed_dim)
        feature_map3[text_offset_index, 1, :300] =  np.mean(np.asarray([token.vector for token in current_sent]),
                                                         axis=0) if len(current_sent) > 0 else np.zeros(embed_dim)
        feature_map3[text_offset_index, 2, :300] =  np.mean(np.asarray([token.vector for token in succeed_sent]),
                                                         axis=0) if len(succeed_sent) > 0 else np.zeros(embed_dim)

        feature_map3[text_offset_index, 0, 300:1324] = sentence_embed(proceed_sents)
        feature_map3[text_offset_index, 1, 300:1324] = sentence_embed(current_sent)
        feature_map3[text_offset_index, 2, 300:1324] = sentence_embed(succeed_sent)

    return feature_map1, feature_map2, feature_map3

extract_embedding_features(table, "Text", "Pronoun-offset")

def extract_dist_features(df, text_column, pronoun_offset_column, name_offset_column):
    text_offset_list = df[[text_column, pronoun_offset_column, name_offset_column]].values.tolist()
    extractor = Distance_Features()
    dist_feas = []

    for text_offset_index in range(len(text_offset_list)):
        text_offset = text_offset_list[text_offset_index]
        dist_fea = extractor.create(text_offset[1], text_offset[2], text_offset[0])
        dist_feas.append(dist_fea)

    return np.asarray(dist_feas)
