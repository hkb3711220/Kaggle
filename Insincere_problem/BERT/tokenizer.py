import pandas as pd
import numpy as np

def split_text(text):

    text = text.strip()
    if not text:
        return []
    tokens = text.split()

    return tokens

def convert_to_index(vocab, tokens):

    output = []
    for token in tokens:
        output.append(vocab[token])

    return output


class fulltokenizer(object):

    def __init__(self, vocab):

        self.vocab = vocab
        self.basictokenizer = basictokenizer(to_lower=True)
        self.wordpiecetokenizer = wordpiecetokenizer(self.vocab)

    def tokenize(self, text):

        split_tokens = []
        text = self.basictokenizer.tokenize(text)
        for tokens in self.wordpiecetokenizer.tokenize(text):
            split_tokens.append(tokens)

        return split_tokens

    def text_to_index(self, tokens):
        return convert_to_index(self.vocab, tokens)

class basictokenizer(object):

    def __init__(self, to_lower=True):

        self.to_lower = to_lower

    def tokenize(self, text):

        tokens = split_text(text)
        if self.to_lower:
            tokens = [token.lower() for token in tokens]

        output = " ".join(tokens)

        return output

class wordpiecetokenizer(object):

    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=200):

        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):

        output_tokens = []
        for token in split_text(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            start = 0
            sub_tokens = []
            is_bad = False
            while start < len(chars):
                end = len(chars)
                cur_str = None
                while start < end:
                    sub_str = "".join(chars[start:end])
                    if start > 0:
                        sub_str = "##" + sub_str
                    if sub_str in self.vocab:
                        cur_str = sub_str
                        break
                    end -=1

                if cur_str is None:
                    is_bad = True
                    break

                sub_tokens.append(cur_str)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens


#For test basictokenizer
#text = "UNWANTED RUNNING"
#text = basictokenizer().tokenize(text)
#print(text)
#For text wordpiecetokenizer
#vocab_tokens =["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing"]
#vocab = {}
#for (i, token) in enumerate(vocab_tokens):
    #vocab[token] = i
#output_tokens = wordpiecetokenizer(vocab).tokenize(text)
#print(output_tokens)
#For text fulltokenizer
#split_tokens = fulltokenizer(vocab).tokenize(text)
#output = fulltokenizer(vocab).text_to_index(split_tokens)
#print(output)
