from tqdm import tqdm
import spacy
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer

nlp = spacy.load('en_core_web_lg')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def candidate_length(candidate):
    count = 0
    for i in range(len(candidate)):
        if candidate[i] !=  " ": count += 1
    return count

def count_char(text, offset):
    count = 0
    for pos in range(offset):
        if text[pos] != " ": count +=1
    return count

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

def count_token_length_special(token):
    count = 0
    special_token = ["#", " "]
    for i in range(len(token)):
        if token[i] not in special_token:
            #print(token[i])
            count+=1
    return count

def find_word_index(tokenized_text, char_start, target):
    tar_len = candidate_length(target)
    char_count = 0
    word_index = []
    special_token = ["[CLS]", "[SEP]"]
    for i in range(len(tokenized_text)):
        token = tokenized_text[i]
        if char_count in range(char_start, char_start+tar_len):
            if token in special_token: # for the case like "[SEP]. she"
                continue
            print(token)
            word_index.append(i)
        if token not in special_token:
            token_length = count_token_length_special(token)
            char_count += token_length

    if len(word_index) == 1:
        return [word_index[0], word_index[0]]
    else:
        return [word_index[0], word_index[-1]]

def create_tokenizer_input(sents):
    tokenizer_input = str()
    for i, sent in enumerate(sents):
        if i == 0:
            tokenizer_input += "[CLS] "+sent.text+" [SEP] "
        elif i == len(sents) - 1:
            tokenizer_input += sent.text+" [SEP]"
        else:
            tokenizer_input += sent.text+" [SEP] "

    return  tokenizer_input

def tokenizer_input(doc, A_offset, B_offset, Pronoun_offset):

    lens = [token.idx for token in doc]
    A_index = bs(lens, A_offset) - 1
    B_index = bs(lens, B_offset) - 1
    Pronoun_index = bs(lens, Pronoun_offset) - 1

    sent_lens = [len(sent) for sent in doc.sents]
    acc_lens  = sent_lens
    pre_lens  = 0
    for i in range(0, len(sent_lens)):
        pre_lens   += acc_lens[i]
        acc_lens[i] = pre_lens
    A_sent_index       = bs(acc_lens, A_index)
    B_sent_index       = bs(acc_lens, B_index)
    Pronoun_sent_index = bs(acc_lens, Pronoun_index)
    sent_set           = sorted(list(set([A_sent_index, B_sent_index, Pronoun_sent_index])))
    sents              = [list(doc.sents)[sent_idx] for sent_idx in sent_set]

    text_length = 0
    previous_sent_last = 0
    current_sent_first = 0
    for i in range(1, len(sent_set)):
        if sent_set[i] - sent_set[i-1] >=2:
            text_length = candidate_length(sents[i-1][-1].text)
            #print(text_length)
            previous_sent_last = sents[i-1][-1].idx
            current_sent_first = sents[i][0].idx
    print(sent_set)
    return create_tokenizer_input(sents), (sents[0][0].idx, previous_sent_last, current_sent_first, text_length)

def create_segment_id(tokenized_text):

    segment_num = 0
    segments_ids = []
    for i in range(len(tokenized_text)):
        token = tokenized_text[i]
        segments_ids.append(segment_num)
        if token == "[SEP]": segment_num += 1

    return segments_ids

def create_inputs(dataframe):

    idxs = dataframe.index
    columns = ['indexed_token', 'offset', 'segment_id']
    features_df = pd.DataFrame(index=idxs, columns=columns)

    for i in tqdm(range(len(dataframe))):

        text           = dataframe.loc[i, 'Text']
        Pronoun_offset = dataframe.loc[i, 'Pronoun-offset']
        A_offset       = dataframe.loc[i, "A-offset"]
        B_offset       = dataframe.loc[i, "B-offset"]
        Pronoun        = dataframe.loc[i, "Pronoun"]
        A              = dataframe.loc[i, "A"]
        B              = dataframe.loc[i, "B"]
        doc            = nlp(text)

        token_input, char_offset = tokenizer_input(doc, A_offset, B_offset, Pronoun_offset)
        token_input = token_input.replace("#", "x")
        tokenized_text = tokenizer.tokenize(token_input)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids   = create_segment_id(tokenized_text)
        print(token_input)

        A_char_start, B_char_start = count_char(text, A_offset), count_char(text, B_offset)
        Pronoun_char_start         = count_char(text, Pronoun_offset)
        first_word_char_start      = count_char(text, char_offset[0])
        gap                        = count_char(text, char_offset[2]) - count_char(text, char_offset[1]) - char_offset[3]
        char_start_set             = np.asarray([A_char_start, B_char_start, Pronoun_char_start]) - first_word_char_start
        word_indexes = []

        for char_start, target in zip(char_start_set, [A, B, Pronoun]):
            if char_start - (count_char(text, char_offset[1]) - first_word_char_start) >= gap and gap != 0:
                char_start = char_start - gap
            word_indexes.append(find_word_index(tokenized_text, char_start, target))
        features_df.iloc[i] = [indexed_tokens, word_indexes, segments_ids]

    return features_df
