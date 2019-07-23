from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
package_dir = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT/"
sys.path.append(package_dir)

import torch.utils.data
import numpy as np
import pandas as pd
import os
import re
import warnings
import time
import random
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import BertConfig
import torch.nn as nn
from nltk.tokenize.treebank import TreebankWordTokenizer
from tqdm import tqdm
warnings.filterwarnings(action='ignore')
device = torch.device('cuda')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class config:
    ##COMMON CONFIG
    PATH = '../input'
    CSV  = {'test': os.path.join(PATH, 'jigsaw-unintended-bias-in-toxicity-classification/test.csv'),
            'sample_submission':os.path.join(PATH, 'jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')}
    SEED = 1234
    BATCH_SIZE = 32
    MAX_SEQUENCE_LENGTH = 220
    ##BERT
    BERT_MODEL_PATH = {'base': '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/',
                       'large': '../input/bert-pretrained-models/uncased_l-24_1024_a-16/uncased_L-24_H-1024_A-16/'}
    BERT_VOCAB = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/vocab.txt'
    BERT_MODEL_WEIGHT    = {'bert_1': os.path.join(PATH, 'bert-test-1'),
                            'bert_2': os.path.join(PATH, 'bert-test-8')}
    
    ##GPT2
    GPT_MODEL_PATH = '../input/gpt2-models/'
    GPT_MODEL_WEIGHT = '../input/gpt2-p1e2-lr10/gpt2_model_lr1.0.bin'
    
    
    ##LSTM
    WORD_INDEX_PATH = '../input/best-rnn/word_index.pk'
    RNN_PATH = '../input/best-rnn/'
    CRAWL_EMBEDDING_PATH = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
    GLOVE_EMBEDDING_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'
    NUM_MODELS = 2
    LSTM_UNITS = 128
    DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
    MAX_LSTM_LENGTH = 300
    MAX_FEATURES = 400000

contraction_mapping = {
    "Trump's" : 'trump is', "trump's" : 'trump is', "'cause": 'because',',cause': 'because',';cause': 'because',"ain't": 'am not','ain,t': 'am not',
    'ain;t': 'am not','ain´t': 'am not','ain’t': 'am not',"aren't": 'are not',
    'aren,t': 'are not','aren;t': 'are not','aren´t': 'are not','aren’t': 'are not',"can't": 'cannot',"can't've": 'cannot have','can,t': 'cannot','can,t,ve': 'cannot have',
    'can;t': 'cannot','can;t;ve': 'cannot have',
    'can´t': 'cannot','can´t´ve': 'cannot have','can’t': 'cannot','can’t’ve': 'cannot have',
    "could've": 'could have','could,ve': 'could have','could;ve': 'could have',"couldn't": 'could not',"couldn't've": 'could not have','couldn,t': 'could not','couldn,t,ve': 'could not have','couldn;t': 'could not',
    'couldn;t;ve': 'could not have','couldn´t': 'could not',
    'couldn´t´ve': 'could not have','couldn’t': 'could not','couldn’t’ve': 'could not have','could´ve': 'could have',
    'could’ve': 'could have',"didn't": 'did not','didn,t': 'did not','didn;t': 'did not','didn´t': 'did not',
    'didn’t': 'did not',"doesn't": 'does not','doesn,t': 'does not','doesn;t': 'does not','doesn´t': 'does not',
    'doesn’t': 'does not',"don't": 'do not','don,t': 'do not','don;t': 'do not','don´t': 'do not','don’t': 'do not',
    "hadn't": 'had not',"hadn't've": 'had not have','hadn,t': 'had not','hadn,t,ve': 'had not have','hadn;t': 'had not',
    'hadn;t;ve': 'had not have','hadn´t': 'had not','hadn´t´ve': 'had not have','hadn’t': 'had not','hadn’t’ve': 'had not have',"hasn't": 'has not','hasn,t': 'has not','hasn;t': 'has not','hasn´t': 'has not','hasn’t': 'has not',
    "haven't": 'have not','haven,t': 'have not','haven;t': 'have not','haven´t': 'have not','haven’t': 'have not',"he'd": 'he would',
    "he'd've": 'he would have',"he'll": 'he will',
    "he's": 'he is','he,d': 'he would','he,d,ve': 'he would have','he,ll': 'he will','he,s': 'he is','he;d': 'he would',
    'he;d;ve': 'he would have','he;ll': 'he will','he;s': 'he is','he´d': 'he would','he´d´ve': 'he would have','he´ll': 'he will',
    'he´s': 'he is','he’d': 'he would','he’d’ve': 'he would have','he’ll': 'he will','he’s': 'he is',"how'd": 'how did',"how'll": 'how will',
    "how's": 'how is','how,d': 'how did','how,ll': 'how will','how,s': 'how is','how;d': 'how did','how;ll': 'how will',
    'how;s': 'how is','how´d': 'how did','how´ll': 'how will','how´s': 'how is','how’d': 'how did','how’ll': 'how will',
    'how’s': 'how is',"i'd": 'i would',"i'll": 'i will',"i'm": 'i am',"i've": 'i have','i,d': 'i would','i,ll': 'i will',
    'i,m': 'i am','i,ve': 'i have','i;d': 'i would','i;ll': 'i will','i;m': 'i am','i;ve': 'i have',"isn't": 'is not',
    'isn,t': 'is not','isn;t': 'is not','isn´t': 'is not','isn’t': 'is not',"it'd": 'it would',"it'll": 'it will',"It's":'it is',
    "it's": 'it is','it,d': 'it would','it,ll': 'it will','it,s': 'it is','it;d': 'it would','it;ll': 'it will','it;s': 'it is','it´d': 'it would','it´ll': 'it will','it´s': 'it is',
    'it’d': 'it would','it’ll': 'it will','it’s': 'it is',
    'i´d': 'i would','I´ll': 'i will','i´m': 'i am','i´ve': 'i have','i’d': 'i would','i’ll': 'i will','i’m': 'i am',
    'i’ve': 'i have',"let's": 'let us','let,s': 'let us','let;s': 'let us','let´s': 'let us',
    'let’s': 'let us',"ma'am": 'madam','ma,am': 'madam','ma;am': 'madam',"mayn't": 'may not','mayn,t': 'may not','mayn;t': 'may not',
    'mayn´t': 'may not','mayn’t': 'may not','ma´am': 'madam','ma’am': 'madam',"might've": 'might have','might,ve': 'might have','might;ve': 'might have',"mightn't": 'might not','mightn,t': 'might not','mightn;t': 'might not','mightn´t': 'might not',
    'mightn’t': 'might not','might´ve': 'might have','might’ve': 'might have',"must've": 'must have','must,ve': 'must have','must;ve': 'must have',
    "mustn't": 'must not','mustn,t': 'must not','mustn;t': 'must not','mustn´t': 'must not','mustn’t': 'must not','must´ve': 'must have',
    'must’ve': 'must have',"needn't": 'need not','needn,t': 'need not','needn;t': 'need not','needn´t': 'need not','needn’t': 'need not',"oughtn't": 'ought not','oughtn,t': 'ought not','oughtn;t': 'ought not',
    'oughtn´t': 'ought not','oughtn’t': 'ought not',"sha'n't": 'shall not','sha,n,t': 'shall not','sha;n;t': 'shall not',"shan't": 'shall not',
    'shan,t': 'shall not','shan;t': 'shall not','shan´t': 'shall not','shan’t': 'shall not','sha´n´t': 'shall not','sha’n’t': 'shall not',
    "she'd": 'she would',"she'll": 'she will',"she's": 'she is','she,d': 'she would','she,ll': 'she will',
    'she,s': 'she is','she;d': 'she would','she;ll': 'she will','she;s': 'she is','she´d': 'she would','she´ll': 'she will',
    'she´s': 'she is','she’d': 'she would','she’ll': 'she will','she’s': 'she is',"should've": 'should have','should,ve': 'should have','should;ve': 'should have',
    "shouldn't": 'should not','shouldn,t': 'should not','shouldn;t': 'should not','shouldn´t': 'should not','shouldn’t': 'should not','should´ve': 'should have',
    'should’ve': 'should have',"that'd": 'that would',"that's": 'that is','that,d': 'that would','that,s': 'that is','that;d': 'that would',
    'that;s': 'that is','that´d': 'that would','that´s': 'that is','that’d': 'that would','that’s': 'that is',"there'd": 'there had',
    "there's": 'there is','there,d': 'there had','there,s': 'there is','there;d': 'there had','there;s': 'there is',
    'there´d': 'there had','there´s': 'there is','there’d': 'there had','there’s': 'there is',
    "they'd": 'they would',"they'll": 'they will',"they're": 'they are',"they've": 'they have',
    'they,d': 'they would','they,ll': 'they will','they,re': 'they are','they,ve': 'they have','they;d': 'they would','they;ll': 'they will','they;re': 'they are',
    'they;ve': 'they have','they´d': 'they would','they´ll': 'they will','they´re': 'they are','they´ve': 'they have','they’d': 'they would','they’ll': 'they will',
    'they’re': 'they are','they’ve': 'they have',"wasn't": 'was not','wasn,t': 'was not','wasn;t': 'was not','wasn´t': 'was not',
    'wasn’t': 'was not',"we'd": 'we would',"we'll": 'we will',"we're": 'we are',"we've": 'we have','we,d': 'we would','we,ll': 'we will',
    'we,re': 'we are','we,ve': 'we have','we;d': 'we would','we;ll': 'we will','we;re': 'we are','we;ve': 'we have',
    "weren't": 'were not','weren,t': 'were not','weren;t': 'were not','weren´t': 'were not','weren’t': 'were not','we´d': 'we would','we´ll': 'we will',
    'we´re': 'we are','we´ve': 'we have','we’d': 'we would','we’ll': 'we will','we’re': 'we are','we’ve': 'we have',"what'll": 'what will',"what're": 'what are',"what's": 'what is',
    "what've": 'what have','what,ll': 'what will','what,re': 'what are','what,s': 'what is','what,ve': 'what have','what;ll': 'what will','what;re': 'what are',
    'what;s': 'what is','what;ve': 'what have','what´ll': 'what will',
    'what´re': 'what are','what´s': 'what is','what´ve': 'what have','what’ll': 'what will','what’re': 'what are','what’s': 'what is',
    'what’ve': 'what have',"where'd": 'where did',"where's": 'where is','where,d': 'where did','where,s': 'where is','where;d': 'where did',
    'where;s': 'where is','where´d': 'where did','where´s': 'where is','where’d': 'where did','where’s': 'where is',
    "who'll": 'who will',"who's": 'who is','who,ll': 'who will','who,s': 'who is','who;ll': 'who will','who;s': 'who is',
    'who´ll': 'who will','who´s': 'who is','who’ll': 'who will','who’s': 'who is',"won't": 'will not','won,t': 'will not','won;t': 'will not',
    'won´t': 'will not','won’t': 'will not',"wouldn't": 'would not','wouldn,t': 'would not','wouldn;t': 'would not','wouldn´t': 'would not',
    'wouldn’t': 'would not',"you'd": 'you would',"you'll": 'you will',"you're": 'you are','you,d': 'you would','you,ll': 'you will',
    'you,re': 'you are','you;d': 'you would','you;ll': 'you will',
    'you;re': 'you are','you´d': 'you would','you´ll': 'you will','you´re': 'you are','you’d': 'you would','you’ll': 'you will','you’re': 'you are',
    '´cause': 'because','’cause': 'because',"you've": "you have","could'nt": 'could not',
    "havn't": 'have not',"here’s": "here is",'i""m': 'i am',"i'am": 'i am',"i'l": "i will","i'v": 'i have',"wan't": 'want',"was'nt": "was not","who'd": "who would",
    "who're": "who are","who've": "who have","why'd": "why would","would've": "would have","y'all": "you all","y'know": "you know","you.i": "you i",
    "your'e": "you are","arn't": "are not","agains't": "against","c'mon": "common","doens't": "does not",'don""t': "do not","dosen't": "does not",
    "dosn't": "does not","shoudn't": "should not","that'll": "that will","there'll": "there will","there're": "there are",
    "this'll": "this all","u're": "you are", "ya'll": "you all","you'r": "you are","you’ve": "you have","d'int": "did not","did'nt": "did not","din't": "did not","dont't": "do not","gov't": "government",
    "i'ma": "i am","is'nt": "is not","‘I":'I',
    'ᴀɴᴅ':'and','ᴛʜᴇ':'the','ʜᴏᴍᴇ':'home','ᴜᴘ':'up','ʙʏ':'by','ᴀᴛ':'at','…and':'and','civilbeat':'civil beat',\
    'TrumpCare':'Trump care','Trumpcare':'Trump care', 'OBAMAcare':'Obama care','ᴄʜᴇᴄᴋ':'check','ғᴏʀ':'for','ᴛʜɪs':'this','ᴄᴏᴍᴘᴜᴛᴇʀ':'computer',\
    'ᴍᴏɴᴛʜ':'month','ᴡᴏʀᴋɪɴɢ':'working','ᴊᴏʙ':'job','ғʀᴏᴍ':'from','Sᴛᴀʀᴛ':'start','gubmit':'submit','CO₂':'carbon dioxide','ғɪʀsᴛ':'first',\
    'ᴇɴᴅ':'end','ᴄᴀɴ':'can','ʜᴀᴠᴇ':'have','ᴛᴏ':'to','ʟɪɴᴋ':'link','ᴏғ':'of','ʜᴏᴜʀʟʏ':'hourly','ᴡᴇᴇᴋ':'week','ᴇɴᴅ':'end','ᴇxᴛʀᴀ':'extra',\
    'Gʀᴇᴀᴛ':'great','sᴛᴜᴅᴇɴᴛs':'student','sᴛᴀʏ':'stay','ᴍᴏᴍs':'mother','ᴏʀ':'or','ᴀɴʏᴏɴᴇ':'anyone','ɴᴇᴇᴅɪɴɢ':'needing','ᴀɴ':'an','ɪɴᴄᴏᴍᴇ':'income',\
    'ʀᴇʟɪᴀʙʟᴇ':'reliable','ғɪʀsᴛ':'first','ʏᴏᴜʀ':'your','sɪɢɴɪɴɢ':'signing','ʙᴏᴛᴛᴏᴍ':'bottom','ғᴏʟʟᴏᴡɪɴɢ':'following','Mᴀᴋᴇ':'make',\
    'ᴄᴏɴɴᴇᴄᴛɪᴏɴ':'connection','ɪɴᴛᴇʀɴᴇᴛ':'internet','financialpost':'financial post', 'ʜaᴠᴇ':' have ', 'ᴄaɴ':' can ', 'Maᴋᴇ':' make ', 'ʀᴇʟɪaʙʟᴇ':' reliable ', 'ɴᴇᴇᴅ':' need ',
    'ᴏɴʟʏ':' only ', 'ᴇxᴛʀa':' extra ', 'aɴ':' an ', 'aɴʏᴏɴᴇ':' anyone ', 'sᴛaʏ':' stay ', 'Sᴛaʀᴛ':' start', 'SHOPO':'shop',
    }

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
small_caps_mapping = {
    "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ғ": "f", "ɢ": "g", "ʜ": "h", "ɪ": "i", 
    "ᴊ": "j", "ᴋ": "k", "ʟ": "l", "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ǫ": "q", "ʀ": "r", 
    "s": "s", "ᴛ": "t", "ᴜ": "u", "ᴠ": "v", "ᴡ": "w", "x": "x", "ʏ": "y", "ᴢ": "z"}
special_signs = { "…": "...", "₂": "2"}

mispell_dict = {'SB91':'senate bill','tRump':'trump','utmterm':'utm term','FakeNews':'fake news','Gʀᴇat':'great','ʙᴏᴛtoᴍ':'bottom','washingtontimes':'washington times','garycrum':'gary crum','htmlutmterm':'html utm term','RangerMC':'car','TFWs':'tuition fee waiver','SJWs':'social justice warrior','Koncerned':'concerned','Vinis':'vinys','Yᴏᴜ':'you','Trumpsters':'trump','Trumpian':'trump','bigly':'big league','Trumpism':'trump','Yoyou':'you','Auwe':'wonder','Drumpf':'trump','utmterm':'utm term','Brexit':'british exit','utilitas':'utilities','ᴀ':'a', '😉':'wink','😂':'joy','😀':'stuck out tongue', 'theguardian':'the guardian','deplorables':'deplorable', 'theglobeandmail':'the globe and mail', 'justiciaries': 'justiciary','creditdation': 'Accreditation','doctrne':'doctrine','fentayal': 'fentanyl','designation-': 'designation','CONartist' : 'con-artist','Mutilitated' : 'Mutilated','Obumblers': 'bumblers','negotiatiations': 'negotiations','dood-': 'dood','irakis' : 'iraki','cooerate': 'cooperate','COx':'cox','racistcomments':'racist comments','envirnmetalists': 'environmentalists',}

# this from other kernel
symbols_to_isolate = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'
symbols_to_delete = '\n🍕\r🐵😑\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'

treetokenizer = TreebankWordTokenizer()

class preprocess_class(object):
    
    def __init__(self, treetokenizer, remove_dict, isolate_dict=None):
        
        self.tokenizer = treetokenizer
        self.isolate_dict = isolate_dict
        self.remove_dict  = remove_dict
        
    def bert_preprocess(self, x):
        
        x = self.correct_spelling(x)
        x = x.lower()
        x = self.clean_contractions(x)
        x = self.clean_special_chars(x, replace_punct=False)
        
        x = self.handle_punctuation(x)
        x = self.handle_contractions(x)
        x = self.fix_quote(x, normal=False)
        
        return x
    
    def normal_preprocess(self, x):
        
        x = self.clean_contractions(x)
        x = self.clean_special_chars(x)
        x = self.correct_spelling(x)
        
        x = self.handle_punctuation(x)
        x = self.handle_contractions(x)
        x = self.fix_quote(x)
        
        return x
    
    def correct_spelling(self, x):
        for word in mispell_dict.keys():
            x = x.replace(word, mispell_dict[word])
            
        return x
    
    def clean_contractions(self, text):
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")
        text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])
        return text
    
    def clean_special_chars(self, text, replace_punct=True):
        for p in punct_mapping:
            text = text.replace(p, punct_mapping[p])
        if replace_punct:
            for p in punct:
                text = text.replace(p, f' {p} ')
        for s in specials:
            text = text.replace(s, specials[s])
        for s in small_caps_mapping:
            text = text.replace(s, small_caps_mapping[s])
        for s in special_signs:
            text = text.replace(s, special_signs[s])
        return text
    
    def handle_punctuation(self, x):
        x = x.translate(self.remove_dict)
        if self.isolate_dict is not None:
            x = x.translate(self.isolate_dict)
        return x
        
    def handle_contractions(self, x):
        return self.tokenizer.tokenize(x)
    
    def fix_quote(self, x, normal=True):
        
        if normal:
            x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
        else:
            x = [x_[1:] if (x_.startswith("'") and x_ != "'s") else x_ for x_ in x]
        x = ' '.join(x)
        
        return x
    
def convert_lines(example, max_seq_length,tokenizer, GPT=False):
    if GPT == False:
        max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in example:
        tokens_a = tokenizer.tokenize(text)
        #print(tokens_a)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        if GPT:
            one_token = tokenizer.convert_tokens_to_ids(tokens_a) + [0]*(max_seq_length - len(tokens_a))
        else:
            one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)

class custom_output_layer(nn.Module):
    
    def __init__(self, input_dim):
        super(custom_output_layer, self).__init__()
        self.linear_out = nn.Linear(input_dim, 1)
        self.linear_aux_out = nn.Linear(input_dim, 6)
        
    def forward(self, x):
        
        result = self.linear_out(x)
        aux_result = self.linear_aux_out(x)
        out = torch.cat([result, aux_result], 1)
        
        return out

def bert_predict(test_df_bert, bert_model_path, trained_weight_path, num_labels, 
                 use_preprocess=True, use_custom_layer=False):
    
    bert_config = BertConfig( os.path.join(trained_weight_path, 'bert_config.json'))
    berttokenizer = BertTokenizer.from_pretrained(bert_model_path, cache_dir=None,do_lower_case=True)
    with open(config.BERT_VOCAB) as f:
        vocab_txt = f.read()
    vocab_txt = vocab_txt.split('\n')
    symbols = symbols_to_delete + symbols_to_isolate
    bert_not_to_delete = []
    bert_to_delete     = []
    for sym in symbols:
        if sym in vocab_txt:
            bert_not_to_delete.append(sym)
        elif ('##'+sym) in vocab_txt:
            bert_not_to_delete.append(sym)
        else:
            bert_to_delete.append(sym)
    remove_dict = {ord(c):f'' for c in bert_to_delete}
    
    if use_preprocess:
        print("preprocessing...")
        preprocessor = preprocess_class(treetokenizer, remove_dict)
        test_df['comment_text'] = test_df['comment_text'].apply(lambda x:preprocessor.bert_preprocess(x))
    test_df['comment_text'] = test_df['comment_text'].astype(str) 
    print("converting word to index...BERT")
    X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), config.MAX_SEQUENCE_LENGTH, berttokenizer)
    
    if use_custom_layer:
        model = BertForSequenceClassification(bert_config, num_labels=1)
        in_features = model.classifier.in_features
        model.classifier = custom_output_layer(in_features)
    else:
        model = BertForSequenceClassification(bert_config, num_labels=num_labels)
        
    model.load_state_dict(torch.load(os.path.join(trained_weight_path, 'bert_pytorch.bin')))
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    test_preds = np.zeros((len(X_test)))
    test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
    test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False)
    tk0 = test_loader
    print("predict start (bert)...")
    for i, (x_batch,) in enumerate(tk0):
        pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
        test_preds[i * 128:(i + 1) * 128] = pred[:, 0].detach().cpu().squeeze().numpy()
    print("predict end (bert)...")    
    test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()
    
    return test_pred

seed_everything(config.SEED)
test_df = pd.read_csv(config.CSV['test'])#.head(10)
start_time = time.time()
print("-----------------------------start bert predict----------------------------------")
BERT_predict = bert_predict(test_df, 
                            bert_model_path = config.BERT_MODEL_PATH['base'],
                            trained_weight_path = config.BERT_MODEL_WEIGHT['bert_1'],
                            num_labels = 8,
                            use_preprocess = False, 
                            use_custom_layer = False)


print("-------------------------------end bert predict----------------------------------")                               
print('BERT spend time:', time.time()-start_time)

print("reset module path...GPT")
sys.path.remove(package_dir)
ppbert_keys = [key for key in sys.modules.keys() if key.startswith('pytorch_pretrained')]
for key in ppbert_keys: sys.modules.pop(key)
os.system('pip install ../input/ppbert-pure/pytorch-pretrained-bert-pure/pytorch-pretrained-BERT-master/ --upgrade')
#sys.path.append(package_dir_a)
print("end reset module path...GPT")

print("reloading test dataset...GPT/LSTM")
print("preprocessing...GPT/LSTM")
isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}
remove_dict = {ord(c):f'' for c in symbols_to_delete}
preprocessor = preprocess_class(treetokenizer, remove_dict, isolate_dict)
test_df = pd.read_csv(config.CSV['test'])#.head(10)
test_df['comment_text'] = test_df['comment_text'].apply(lambda x:preprocessor.normal_preprocess(x))

print("-----------------------------start GPT2 predict----------------------------------")
start_time = time.time()
from pytorch_pretrained_bert.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Config, OpenAIAdam
print(sys.modules['pytorch_pretrained_bert'])




class GPT2ClassificationHeadModel(GPT2PreTrainedModel):

    def __init__(self, config, clf_dropout=0.4, n_class=8):
        super(GPT2ClassificationHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.dropout = nn.Dropout(clf_dropout)
        self.linear = nn.Linear(config.n_embd * 2, n_class)

        nn.init.normal_(self.linear.weight, std = 0.02)
        nn.init.normal_(self.linear.bias, 0)
        
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        avg_pool = torch.mean(hidden_states, 1)
        max_pool, _ = torch.max(hidden_states, 1)
        h_conc = torch.cat((avg_pool, max_pool), 1)
        logits = self.linear(self.dropout(h_conc))
        return logits

Gptconfig = GPT2Config(config.GPT_MODEL_PATH +'config.json')
Gpttokenizer = GPT2Tokenizer.from_pretrained(config.GPT_MODEL_PATH)
model = GPT2ClassificationHeadModel(Gptconfig, clf_dropout=0.4, n_class=8)
model.load_state_dict(torch.load(config.GPT_MODEL_WEIGHT))
print("converting word to index...GPT")
X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), 
                       config.MAX_SEQUENCE_LENGTH, 
                       Gpttokenizer, 
                       GPT=True)

model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()
batch_size = 64
test_preds = np.zeros((len(X_test)))
print("predict start (GPT2)...")
test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
for i, (x_batch,) in enumerate(test_loader):
    pred = model(x_batch.to(device))
    test_preds[i * batch_size:(i + 1) * batch_size] = pred[:, 0].detach().cpu().squeeze().numpy()
GPT_predict = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()
print("predict end (GPT2)...")
print("-------------------------------end GPT2 predict----------------------------------")
print('GPT spend time:', time.time()-start_time)

print("-----------------------------start LSTM predict----------------------------------")
start_time = time.time()
import fastai
from fastai.train import Learner
from fastai.train import DataBunch
from fastai.callbacks import *
from fastai.basic_data import DatasetType
import fastprogress
from fastprogress import force_console_behavior
from pprint import pprint
from keras.preprocessing import text, sequence


with open(config.WORD_INDEX_PATH, 'rb') as f:
    word_index = pickle.load(f)

def load_embeddings(path):
    with open(path,'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((config.MAX_FEATURES + 1, 300))
    unknown_words = []
    
    for word, i in word_index.items():
        if i <= config.MAX_FEATURES:
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                try:
                    embedding_matrix[i] = embedding_index[word.lower()]
                except KeyError:
                    try:
                        embedding_matrix[i] = embedding_index[word.title()]
                    except KeyError:
                        unknown_words.append(word)
    return embedding_matrix, unknown_words

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class NeuralNet(nn.Module):
    
    def __init__(self, embedding_matrix, num_aux_targets):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, config.LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.GRU(config.LSTM_UNITS * 2, config.LSTM_UNITS, bidirectional=True, batch_first=True)
    
        self.linear1 = nn.Linear(config.DENSE_HIDDEN_UNITS, config.DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(config.DENSE_HIDDEN_UNITS, config.DENSE_HIDDEN_UNITS)
        
        self.linear_out = nn.Linear(config.DENSE_HIDDEN_UNITS, 1)
        self.linear_aux_out = nn.Linear(config.DENSE_HIDDEN_UNITS, num_aux_targets)
        
    def forward(self, x, lengths=None):
        h_embedding = self.embedding(x.long())
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        
        return out
        
    def save_my_weights(self, filename):
        my_weights = deepcopy(self.state_dict())
        my_weights.pop('embedding.weight')
        with open(filename, 'wb') as f:
            pickle.dump(my_weights, f)
        del my_weights
            
    def load_my_weights(self, filename):
        with open(filename, 'rb') as f:
            weights_dict = pickle.load(f)
        weights_dict['embedding.weight'] = self.embedding.weight
        self.load_state_dict(weights_dict)
        #print(self.state_dict())

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SequenceBucketCollator():
    def __init__(self, choose_length, sequence_index, length_index, label_index=None):
        self.choose_length = choose_length
        self.sequence_index = sequence_index
        self.length_index = length_index
        self.label_index = label_index
        
    def __call__(self, batch):
        batch = [torch.stack(x) for x in list(zip(*batch))]
        
        sequences = batch[self.sequence_index]
        lengths = batch[self.length_index]
        
        length = self.choose_length(lengths)
        mask = torch.arange(start=config.MAX_LSTM_LENGTH, end=0, step=-1) < length
        padded_sequences = sequences[:, mask]
        
        batch[self.sequence_index] = padded_sequences
        
        if self.label_index is not None:
            return [x for i, x in enumerate(batch) if i != self.label_index], batch[self.label_index]
    
        return batch

x_test = test_df['comment_text']
lstmtokenizer = text.Tokenizer(num_words = config.MAX_FEATURES, filters='',lower=False)
lstmtokenizer.word_index = word_index
crawl_matrix, unknown_words_crawl = build_matrix(lstmtokenizer.word_index, config.CRAWL_EMBEDDING_PATH)
print('n unknown words (crawl): ', len(unknown_words_crawl))

glove_matrix, unknown_words_glove = build_matrix(lstmtokenizer.word_index, config.GLOVE_EMBEDDING_PATH)
print('n unknown words (glove): ', len(unknown_words_glove))

max_features = config.MAX_FEATURES or len(lstmtokenizer.word_index) + 1

embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)
print('n embedding matrix shape: ', embedding_matrix.shape)

x_test = lstmtokenizer.texts_to_sequences(x_test)

del crawl_matrix
del glove_matrix
gc.collect()

test_lengths = torch.from_numpy(np.array([len(x) for x in x_test]))
x_test_padded = torch.from_numpy(sequence.pad_sequences(x_test, maxlen=config.MAX_LSTM_LENGTH))
batch_size = 512
test_dataset = torch.utils.data.TensorDataset(x_test_padded, test_lengths)

def predict_from_model(model_path,test,output_dim,
                       batch_size=512, 
                       n_epochs=5,
                       enable_checkpoint_ensemble=True):
    
    all_test_preds = []
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    for epoch in range(n_epochs):
        test_preds = np.zeros((len(test), output_dim))
        model = NeuralNet(embedding_matrix, 6)
        model.to(device)
        model_name = model_path + 'model_' + str(model_idx) + '_epoch_' + str(epoch) + '.bin'
        model.load_my_weights(model_name)
        for param in model.parameters(): 
            param.requires_grad = False
        model.eval()
        print('loaded model of', model_name)
        
        for i, x_batch in enumerate(test_loader):
            X = x_batch[0].cuda()
            y_pred = sigmoid(model(X).detach().cpu().numpy())
            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred
            
        all_test_preds.append(test_preds)

    if enable_checkpoint_ensemble:
        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)    
    else:
        test_preds = all_test_preds[-1]
        
    return test_preds


all_test_preds = []
for model_idx in range(config.NUM_MODELS):
    print('Model ', model_idx)
    seed_everything(1 + model_idx)
    test_preds = predict_from_model(config.RNN_PATH,test_dataset,output_dim=7) #, test_collator=test_collator)  
    all_test_preds.append(test_preds)

LSTM_predict = np.mean(all_test_preds, axis=0)[:, 0]
print("-----------------------------end LSTM predict----------------------------------")
print('LSTM spend time:', time.time()-start_time)


#-----------------------------BLEND----------------------------------
#from sklearn.metrics import mean_squared_error
#import math
#submission1 = pd.read_csv("../input/blend-test/submission_GPT.csv")
#submission2 = pd.read_csv("../input/blend-test/submission_BERT.csv")
#submission3 = pd.read_csv("../input/best-rnn//submission.csv")

#error1 = math.sqrt(mean_squared_error(submission2.prediction, BERT_predict))
#print('Bert Inference error:', error1)

#error2 = math.sqrt(mean_squared_error(submission1.prediction, GPT_predict))
#print('GPT Inference error:', error2)

#error3 = math.sqrt(mean_squared_error(submission3.prediction, LSTM_predict))
#print('LSTM Inference error:', error3)

FINAL_ANSWER = 0.7 * (0.2*LSTM_predict + 0.8*BERT_predict) + 0.3 * GPT_predict
submission = pd.DataFrame.from_dict({
    'id': test_df['id'],
    'prediction': FINAL_ANSWER})

submission.to_csv('submission.csv', index=False)
