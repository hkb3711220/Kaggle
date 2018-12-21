import sys
sys.path.append('./')

import pandas as pd
import numpy as np
import random
import os
from tokenizer import *

def create_pretrain_data(input_file, vocab, tokenizer, max_sequence_length, rng):

    all_documents = []
    with open(input_file) as f:
        for sequence in f.readlines()[:2]:
            sequence = sequence.strip()
            tokens = tokenizer.tokenize(sequence)
            if tokens:
                all_documents.append(tokens)

    #remove empty document
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_keys = list(vocab.keys())
    instances = []

    for documents_index in range(len(all_documents)):
        document = all_documents[documents_index]
        instances.append(create_instances_from_document(document,
                                                        vocab_keys,
                                                        max_sequence_length,rng))

    return instances

def create_instances_from_document(document, vocab, max_sequence_length, rng):
    """
    For single-sentence inputs we only use the sentence A embeddings
    """
    tokens_output = []
    tokens_output.append('[CLS]')
    for tokens in document:
        tokens_output.append(tokens)

    tokens_output.append('[SEP]')
    tokens_output = make_mask(tokens_output, vocab, rng, mask_prob=0.1)

    return tokens_output

def make_mask(tokens, vocab, rng, mask_prob,
              max_predictions_per_seq=20):

    cand_index = []

    for i, token in enumerate(tokens):
        if token == '[CLS]' or token == '[SEP]':
            continue
        cand_index.append(i)

    rng.shuffle(cand_index)
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens)*mask_prob))))
    #print(num_to_predict)

    output_tokens = list(tokens)
    masked_label = []
    masked_index = []

    for index in cand_index:
        if len(masked_index) >= num_to_predict:
            break

        masked_token = None
        if rng.random() <0.8:
            mask_token = ['MASK']
        else:
            #keep original
            if rng.random() < 0.5:
                mask_token =  tokens[index]
            else:
                mask_token = vocab[rng.randint(0, len(vocab) -1)]

        output_tokens[index] = mask_token
        masked_index.append(index)

    masked_index.sort()
    for index in masked_index:
        masked_label.append(tokens[index])

    return output_tokens


path = r'C:\Users\user1\Desktop\Kaggle\Insincere_problem\BERT\Input'
input_file = os.path.join(path, 'small_vocab_en')

vocab_file = {}
vocab_list = ["[UNK]", "[CLS]", "[SEP]", "new", "jersey", "is", "sometime", "##s","quiet", "during", "autumn", "and", "it" , "snow", "##y", "in" , "april", "the" ,"united", "states", "is", "usually", "chill", "during", "july", "usually", "freez", "##ing", "november"]
for i, vocab in enumerate(vocab_list):
    vocab_file[vocab] = i
#print(vocab_file)
Tokenizer = fulltokenizer(vocab_file)
rng = random.Random(12345)
create_pretrain_data(input_file, vocab_file, Tokenizer, 20, rng)
