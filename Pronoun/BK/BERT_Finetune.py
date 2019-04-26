import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_lg')

table = pd.read_table('./test_stage_1.tsv')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = table['Text'][0]
P_offset = table["Pronoun-offset"][0]
doc = nlp(text)
def count_char(text, offset):
    count = 0
    for pos in range(offset):
        if text[pos] != " ": count +=1
    return count
P_char_start = count_char(text, P_offset)

def create_token_input(doc):
    token_input = str()
    for i, sent in enumerate(doc.sents):
        if i == 0:
            token_input += "[CLS] "+sent.text+" [SEP] "
        elif i == len(list(doc.sents)) - 1:
            token_input += sent.text+" [SEP]"
        else:
            token_input += sent.text+" [SEP] "
    return  token_input

token_input = create_token_input(doc)
tokenized_text = tokenizer.tokenize(token_input)


def find_mask(tokenized_text, char_start):

    char_count = 0
    speical_token = ["[CLS]", "[SEP]"]
    special_symbol = ["#",  " "]
    for i in range(len(tokenized_text)):
        if char_count == char_start: return i
        token = tokenized_text[i]
        if token not in speical_token:
            count = 0
            for i in range(len(token)):
                if token[i] not in special_symbol:
                    count +=1
            char_count += count

masked_index = find_mask(tokenized_text, P_char_start)
tokenized_text[masked_index] = "[MASK]"
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

def create_segment_id(tokenized_text):

    segment_num = 0
    segments_ids = []
    for i in range(len(tokenized_text)):
        token = tokenized_text[i]
        segments_ids.append(segment_num)
        if token == "[SEP]": segment_num += 1

    return segments_ids

#segments_ids = create_segment_id(tokenized_text)

tokens_tensor = 	torch.LongTensor([indexed_tokens])
#segments_tensors = torch.LongTensor([segments_ids])

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

with torch.no_grad():
    predictions = model(tokens_tensor)

print(predictions)
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

#print(predicted_token)
#print(predicted_token)