import pandas as pd
import os
import nltk
from nltk import word_tokenize
import nltk
import spacy
nlp_model = spacy.load('en_core_web_sm')
from spacy.lang.en import English
from nltk import Tree

os.chdir(os.path.dirname(__file__))

table = pd.read_table('./test_stage_1.tsv')
table.to_csv('test_stage_1.csv', index = False)

def word_tags(text):
    words = word_tokenize(text)
    tags  = nltk.pos_tag(words)
    pronoun =[a for (a,b) in tags if a == 'her']

    return pronoun

print(word_tags(table['Text'][0]))
doc = nlp_model(table['Text'][0])

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

#[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

for s in doc.sents:
    print(s)

print(doc.speaker(1))#displacy.render(doc, style='dep', options={'distance': 90})
