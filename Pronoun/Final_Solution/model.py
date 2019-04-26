import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
from tqdm import tqdm_notebook as tqdm
from keras.layers import *
import keras.backend as K
from keras.models import *
import keras
from keras import optimizers
from keras import callbacks
from keras.initializers import he_normal
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

def _to_index(dataframe):
    
    for cols in ['A_dep', 'B_dep', 'Pronoun_dep']:
        dataframe[cols] = dataframe[cols].map(dep_dict)
        
    return dataframe

def dep_to_index(train_df, test_df, save=False):
    global x
    x = pd.concat([train_df, test_df])
    global all_dep
    all_dep = list(set(list(x['A_dep']) + list(x['B_dep']) + list(x['Pronoun_dep'])))
    
    global dep_dict
    dep_dict = {}
    for i in range(len(all_dep)):
        dep_dict[all_dep[i]] = i+1
    dep_dict['unk'] = 0
    
    if save:
        with open('%s.dump'%("dep"), 'wb') as f:
            pickle.dump(dep_dict, f)
            
    train_df = _to_index(train_df)
    test_df = _to_index(test_df)
    
    return train_df, test_df

def min_max(x, min_, max_, range_needed=(0, 1)):
    norm = (x+abs(min_))/(max_+abs(min_))
    return norm * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    
def norm_to_index(dataframe):
    
    for col in ['A_count', 'B_count', 'Pronoun_count']:
        dataframe[col] =  min_max(dataframe[col].values, min_count, max_count) 
        
    return dataframe

def normlize_pair_count(train_df, test_df):
    
    all_pair = list(set(list(x['A/P']) + list(x['B/P'])))
    max_num = max(all_pair)
    min_num = min(all_pair)
    
    train_df['A/P'] = min_max(train_df['A/P'].values, min_num, max_num) 
    train_df['B/P'] = min_max(train_df['B/P'].values, min_num, max_num)
    test_df['A/P']  = min_max(test_df['A/P'].values, min_num, max_num) 
    test_df['B/P']  = min_max(test_df['B/P'].values, min_num, max_num)
    
    return train_df, test_df

def normlize_count(train_df, test_df):
    
    all_count = list(set(list(x['A_count']) + list(x['B_count']) + list(x['Pronoun_count'])))
    global max_count
    global min_count
    max_count = max(all_count)
    min_count = min(all_count)
    
    train_df = norm_to_index(train_df)
    test_df = norm_to_index(test_df)
    
    return train_df, test_df


def create_inputs(embeddings, features, training=True):

    # Embedings
    A_embed = np.zeros((len(embeddings), 1024))
    B_embed = np.zeros((len(embeddings), 1024))
    Pronoun_embed = np.zeros((len(embeddings), 1024))

    # Pair Features
    dist_pa = np.zeros((len(embeddings), 1))
    dist_pb = np.zeros((len(embeddings), 1))
    sent_pa = np.zeros((len(embeddings), 1))
    sent_pb = np.zeros((len(embeddings), 1))
    pair_ap = np.zeros((len(embeddings), 3))
    pair_bp = np.zeros((len(embeddings), 3))
    # Singleton Features
    posa, posb, posp = np.zeros((len(embeddings), 1)), np.zeros((len(embeddings), 1)), np.zeros((len(embeddings), 1))
    a_dep, b_dep, p_dep = np.zeros((len(embeddings), 1)), np.zeros((len(embeddings), 1)), np.zeros((len(embeddings), 1))

    a_singleton_features = np.zeros((len(embeddings), 6))
    b_singleton_features = np.zeros((len(embeddings), 6))
    p_singleton_features = np.zeros((len(embeddings), 6))

    Y = []
    # Concatenate features
    for i in tqdm(range(len(embeddings))):
        A_embed[i, :] = np.array(embeddings.loc[i, "emb_A"])
        B_embed[i, :] = np.array(embeddings.loc[i, "emb_B"])
        Pronoun_embed[i, :] = np.array(embeddings.loc[i, "emb_P"])

        #Pair Features
        dist_pa[i, :] = features.loc[i, "distpa"]
        dist_pb[i, :] = features.loc[i, "distpb"]
        sent_pa[i, :] = features.loc[i, "sent_distpa"]
        sent_pb[i, :] = features.loc[i, "sent_distpb"]
        pair_ap[i, :] = np.asarray([features.loc[i, "in_A_sent"], features.loc[i, "before_pa"], features.loc[i, "A/P"]])
        pair_bp[i, :] = np.asarray([features.loc[i, "in_B_sent"], features.loc[i, "before_pb"], features.loc[i, "B/P"]])

        #Singleton Features
        a_dep[i, :] = features.loc[i, "A_dep"]
        b_dep[i, :] = features.loc[i, "B_dep"]
        p_dep[i, :] = features.loc[i, "Pronoun_dep"]
        posa[i, :]  = features.loc[i, "pos_a"]
        posb[i, :]  = features.loc[i, "pos_b"]
        posp[i, :]  = features.loc[i, "pos_pronoun"]

        a_singleton_features[i, :] = np.asarray([
                                                 features.loc[i, "A_is_plural"],
                                                 features.loc[i, "A_is_object"],
                                                 features.loc[i, "A_on_url"],
                                                 features.loc[i, "A_is_locative"],
                                                 features.loc[i, "A_is_female"],
                                                 features.loc[i, "A_is_male"]], dtype=np.float32)

        b_singleton_features[i, :] = np.asarray([
                                                 features.loc[i, "B_is_plural"],
                                                 features.loc[i, "B_is_object"],
                                                 features.loc[i, "B_on_url"],
                                                 features.loc[i, "B_is_locative"],
                                                 features.loc[i, "B_is_female"],
                                                 features.loc[i, "B_is_male"]], dtype=np.float32)

        p_singleton_features[i, :] = np.asarray([
                                                 features.loc[i, "Pronoun_is_plural"],
                                                 features.loc[i, "Pronoun_is_object"],
                                                 features.loc[i, "Pronoun_on_url"],
                                                 features.loc[i, "Pronoun_is_locative"],
                                                 features.loc[i, "Pronoun_is_female"],
                                                 features.loc[i, "Pronoun_is_male"]], dtype=np.float32)

    # There are some nan embeddings, pick up them and pad them to 0
    A_embed[np.isnan(A_embed)] = 0
    B_embed[np.isnan(B_embed)] = 0
    Pronoun_embed[np.isnan(Pronoun_embed)] = 0

    A_embed = np.concatenate([A_embed, a_singleton_features], axis=1)
    B_embed = np.concatenate([B_embed, b_singleton_features], axis=1)
    Pronoun_embed = np.concatenate([Pronoun_embed, p_singleton_features], axis=1)

    if training:
        for i in range(len(embeddings)):
            label = embeddings.loc[i, "label"]
            if label == "A":
                Y.append(0)
            elif label == "B":
                Y.append(1)
            else:
                Y.append(2)

        return [A_embed, B_embed, Pronoun_embed, posa, posb, posp, a_dep, b_dep, p_dep,  dist_pa, dist_pb, sent_pa, sent_pb, pair_ap, pair_bp], Y
    else:
        return [A_embed, B_embed, Pronoun_embed, posa, posb, posp, a_dep, b_dep, p_dep,  dist_pa, dist_pb, sent_pa, sent_pb, pair_ap, pair_bp]


class Score_model():

    def __init__(self, word_input_shape):
        
        self.word_input_shape = word_input_shape
        self.all_dep      = all_dep
        self.buckets      = [1, 2, 3, 4, 5, 8, 16, 32, 64]
        self.pos_buckets  = [0, 1, 2, 3, 4, 5, 8, 16, 32]
        self.sent_buckets = [0, 1, 2, 3, 4, 5]

    def build(self):
        A, B, P = Input((self.word_input_shape,)), Input((self.word_input_shape,)), Input((self.word_input_shape,))
        # pair Features
        dist1, dist2           = Input((1,)), Input((1,))
        sent_dist1, sent_dist2 = Input((1,)), Input((1,))
        pairap, pairbp         = Input((3,)), Input((3,))
        # singleton Features
        posA, posB, posP  = Input((1,)), Input((1,)), Input((1,))
        depA, depB, depP  = Input((1,)), Input((1,)), Input((1,))

        word_inputs = [A, B, P] #word embed with singleton features
        dist_inputs = [dist1, dist2]
        sent_inputs = [sent_dist1, sent_dist2]
        pair_inputs = [pairap, pairbp]
        pos_inputs  = [posA, posB, posP]
        dep_inputs  = [depA, depB, depP]

        # Define Layers
        self.dist_embed = Embedding(len(self.buckets) + 1, len(self.buckets))
        self.sentdist_embed = Embedding(len(self.sent_buckets)+1, len(self.sent_buckets))
        self.pos_embed = Embedding(len(self.pos_buckets) + 1, len(self.pos_buckets))
        self.dep_embed = Embedding(len(self.all_dep) + 1, len(self.all_dep))
        self.drop = Dropout(0.45)
        self.ffnn = Sequential([Dense(250, use_bias=True, kernel_initializer='he_normal'),
                                      BatchNormalization(),
                                      Activation('relu'),
                                      Dropout(rate=0.79),
                                      Dense(100, use_bias=True, kernel_initializer='he_normal'),
                                      BatchNormalization(),
                                      Activation('relu'),
                                      Dropout(rate=0.0),
                                      Dense(1, activation='linear')])
        self.dot = Dot(axes=-1, normalize=False)

        # Embedding Layer
        dist_embeds = [self.dist_embed(dist) for dist in dist_inputs]
        dist_embeds = [Flatten()(dist_embed) for dist_embed in dist_embeds]

        sent_embeds = [self.sentdist_embed(sent) for sent in sent_inputs]
        sent_embeds = [Flatten()(sent_embed) for sent_embed in sent_embeds]

        pos_embeds = [self.pos_embed(pos) for pos in pos_inputs]
        pos_embeds = [Flatten()(pos_embed) for pos_embed in pos_embeds]
        
        dep_embeds = [self.dep_embed(dep) for dep in dep_inputs]
        dep_embeds = [Flatten()(dep_embed) for dep_embed in dep_embeds]


        reforced_inputs = []
        for i, inp in enumerate(word_inputs):
            reforced_inputs.append(Concatenate(axis=-1)([inp, pos_embeds[i], dep_embeds[i]]))

        # Score layer:
        # the basic idea from https://www.aclweb.org/anthology/D17-1018,
        # used feed forward network which measures if it is an entity mention using a score
        # In here, focus on pairwise score only
        # PairScore: sa(i,j) =wa·FFNNa([gi,gj,gi◦gj,φ(i,j)])
        # gi is embedding of Pronoun
        # gj is embedding of A or B
        # gi◦gj is element-wise multiplication
        # φ(i,j) is the pair features
        pa_multi = Multiply()([reforced_inputs[0], reforced_inputs[2]])
        pb_multi = Multiply()([reforced_inputs[1], reforced_inputs[2]])
        pa_dot = self.dot([reforced_inputs[0], reforced_inputs[2]])
        pb_dot = self.dot([reforced_inputs[1], reforced_inputs[2]])

        PA = Concatenate(axis=-1)([reforced_inputs[2], reforced_inputs[0], pa_multi, dist_embeds[0], sent_embeds[0], pairap, pa_dot])
        PB = Concatenate(axis=-1)([reforced_inputs[2], reforced_inputs[1], pb_multi, dist_embeds[1], sent_embeds[1], pairbp, pb_dot])
        PA = self.drop(PA)
        PB = self.drop(PB)
        PA_score = self.ffnn(PA)
        PB_score = self.ffnn(PB)
        # Fix the Neither to score 0.
        score_e = Lambda(lambda x: K.zeros_like(x))(PB_score)

        # Final Output
        output = Concatenate(axis=-1)([PA_score, PB_score, score_e])  # [Pronoun and A score, Pronoun and B score, Neither Score]
        output = Activation('softmax')(output)
        model = Model(word_inputs + pos_inputs + dep_inputs + dist_inputs + sent_inputs + pair_inputs, output)

        return model

if __name__ == "__main__":
    
    STAGE_1_BERT_PATH = "../input/large-bert-output/large_bert_output/large_bert_output"
    STAGE_1_HANDMADE_OUTPUT = "../input/large-bert-full-comment-large-is-good-model1"
    STAGE_2_BERT_PATH = "../input/bert-embedding"
    STAGE_2_HANDMADE_OUTPUT = "../input/feature-extractor"
    
    stage_1_train_features = pd.read_csv(os.path.join(STAGE_1_HANDMADE_OUTPUT, 'train_handmade.csv'))
    stage_1_test_features  = pd.read_csv(os.path.join(STAGE_1_HANDMADE_OUTPUT, 'test_handmade.csv'))
    stage_1_val_features   = pd.read_csv(os.path.join(STAGE_1_HANDMADE_OUTPUT, 'val_handmade.csv'))
    stage_1_test_embed  = pd.read_json(os.path.join(STAGE_1_BERT_PATH, 'contextual_embeddings_gap_test.json'))
    stage_1_test_embed.sort_index(inplace = True)
    stage_1_val_embed   = pd.read_json(os.path.join(STAGE_1_BERT_PATH, 'contextual_embeddings_gap_val.json'))
    stage_1_val_embed.sort_index(inplace = True)
    stage_1_train_embed = pd.read_json(os.path.join(STAGE_1_BERT_PATH, 'contextual_embeddings_gap_train.json'))
    stage_1_train_embed.sort_index(inplace = True)
    
    #stage2 
    stage_2_embed = pd.read_json(os.path.join(STAGE_2_BERT_PATH, 'contextual_embeddings_stage_0.json'))
    stage_2_embed.sort_index(inplace = True)
    for i in range(12):
        temp_embed = pd.read_json(os.path.join(STAGE_2_BERT_PATH, 'contextual_embeddings_stage_{}.json'.format(i+1)))
        temp_embed.sort_index(inplace = True)
        stage_2_embed = pd.concat([stage_2_embed, temp_embed])
    stage_2_embed = stage_2_embed.reset_index(drop=True)
    test_features = pd.read_csv(os.path.join(STAGE_2_HANDMADE_OUTPUT, 'stage2_handmade.csv'))
    
    train_embed = pd.concat([stage_1_train_embed, stage_1_test_embed, stage_1_val_embed])
    train_embed = train_embed.reset_index(drop=True)
    train_features = pd.concat([stage_1_train_features, stage_1_test_features, stage_1_val_features])
    train_features = train_features.reset_index(drop=True)
    test_embed    = stage_2_embed
    
    train_features, test_features = dep_to_index(train_features, test_features)
    train_features, test_features = normlize_pair_count(train_features, test_features)
    
    X_train, y_train = create_inputs(train_embed, train_features)
    X_test   = create_inputs(test_embed, test_features, training=False)
    
    model = Score_model(word_input_shape=X_train[0].shape[1]).build()
    model.summary()
    min_loss = 100.0
    n_fold = 5
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=2019)
    for fold_n, (train_index, valid_index) in enumerate(kfold.split(X_train[0], y_train)):
        X_tr  = [inputs[train_index] for inputs in X_train]
        X_val = [inputs[valid_index] for inputs in X_train]
        y_tr  = np.asarray(y_train)[train_index]
        y_val = np.asarray(y_train)[valid_index]
        model = Score_model(word_input_shape=X_train[0].shape[1]).build()
        model.compile(optimizer=optimizers.Adam(lr=1e-3, decay=3e-9, beta_1=0, beta_2=0.95), loss="sparse_categorical_crossentropy")
        file_path = "best_model_{}.hdf5".format(fold_n+1)
        check_point = callbacks.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 0, save_best_only = True, mode = "min")
        early_stop = callbacks.EarlyStopping(monitor = "val_loss", mode = "min", patience=100)
        hist = model.fit(X_tr, y_tr, batch_size=20, epochs=1000, validation_data=(X_val, y_val), verbose=0,
                         shuffle=True, callbacks = [check_point, early_stop])
                         
        print(fold_n+1, min(hist.history['val_loss']))
        if min(hist.history['val_loss']) < min_loss:
            min_loss = min(hist.history['val_loss'])
            best_model = fold_n + 1
     
    # use kfold to predict  
    pred = np.zeros((12359, 3))
    for i in range(n_fold):
        model = Score_model(word_input_shape=X_train[0].shape[1]).build()
        model.load_weights("./best_model_{}.hdf5".format(i+1))
        pred += model.predict(x = X_test, batch_size=128, verbose = 0)
    
    pred /= n_fold
    sub_df_path = os.path.join('../input/gendered-pronoun-resolution/', 'sample_submission_stage_2.csv')
    sub_df = pd.read_csv(sub_df_path)
    sub_df.loc[:, 'A'] = pd.Series(pred[:, 0])
    sub_df.loc[:, 'B'] = pd.Series(pred[:, 1])
    sub_df.loc[:, 'NEITHER'] = pd.Series(pred[:, 2])
    sub_df.to_csv("submission.csv", index=False)
    #y_one_hot = np.zeros((2000, 3))
    #_pred = np.argmax(pred, axis=1)
    #target_names = ['A', 'B', 'Neither']
    #for i in range(len(y_test)):
        #y_one_hot[i, y_test[i]] = 1
    #print('loss kfold:', log_loss(y_one_hot, pred))
    #print('classfication_report:\n',classification_report(y_test, _pred, target_names=target_names))
    
    #Use single model
    #model = Score_model(word_input_shape=X_train[0].shape[1]).build()
    #model.load_weights("./best_model_{}.hdf5".format(best_model))
    #pred_single_model = model.predict(x = X_test, verbose = 0)
    #print('loss single model:', log_loss(y_one_hot, pred_single_model))
    #pred_single_ = np.argmax(pred_single_model, axis=1)
    #print('classfication_report:\n',classification_report(y_test, pred_single_, target_names=target_names))
    
    
    
    
    

