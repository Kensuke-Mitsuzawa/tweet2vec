#! -*- coding: utf-8 -*-
# module
import tweet2vec.batch_char as batch
from tweet2vec.t2v import tweet2vec, init_params, load_params
from tweet2vec.settings_char import N_BATCH, MAX_LENGTH, MAX_CLASSES
from tweet2vec import evaluate
# theano
import numpy as np
import lasagne
import theano
import theano.tensor as T
# logger
from tweet2vec.logger import logger
# else
import random
import sys
import time
import io
from collections import OrderedDict
import six
if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl


def classify(tweet, t_mask, params, n_classes, n_chars):
    # tweet embedding
    emb_layer = tweet2vec(tweet, t_mask, params, n_chars)
    # Dense layer for classes
    l_dense = lasagne.layers.DenseLayer(emb_layer, n_classes, W=params['W_cl'], b=params['b_cl'], nonlinearity=lasagne.nonlinearities.softmax)

    return lasagne.layers.get_output(l_dense), lasagne.layers.get_output(emb_layer)


def main(args, is_save=True):

    data_path = args[0]
    model_path = args[1]
    save_path = args[2]
    if len(args)>3:
        m_num = int(args[3])

    logger.debug("Preparing Data...")
    # Test data
    Xt = []
    yt = []
    with io.open(data_path,'r',encoding='utf-8') as f:
        for line in f:
            (yc, Xc) = line.rstrip('\n').split('\t')
            Xt.append(Xc[:MAX_LENGTH])
            yt.append(yc.split(','))

    # Model
    logger.debug("Loading model params...")
    if len(args)>3:
        params = load_params('%s/model_%d.npz' % (model_path,m_num))
    else:
        params = load_params('%s/best_model.npz' % model_path)

    logger.debug("Loading dictionaries...")
    with open('%s/dict.pkl' % model_path, 'rb') as f:
        chardict = pkl.load(f)
    with open('%s/label_dict.pkl' % model_path, 'rb') as f:
        labeldict = pkl.load(f)
    n_char = len(chardict.keys()) + 1
    n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)

    # iterators
    test_iter = batch.BatchTweets(Xt, yt, labeldict, batch_size=N_BATCH, max_classes=MAX_CLASSES, test=True)

    logger.debug("Building network...")
    # Tweet variables
    tweet = T.itensor3()
    targets = T.imatrix()

    # masks
    t_mask = T.fmatrix()

    # network for prediction
    predictions = classify(tweet, t_mask, params, n_classes, n_char)[0]
    embeddings = classify(tweet, t_mask, params, n_classes, n_char)[1]

    # Theano function
    logger.debug("Compiling theano functions...")
    predict = theano.function([tweet,t_mask],predictions)
    encode = theano.function([tweet,t_mask],embeddings)

    # Test
    logger.debug("Testing...")
    out_data = []
    out_pred = []
    out_emb = []
    out_target = []
    for xr,y in test_iter:
        x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
        p = predict(x,x_m)
        e = encode(x,x_m)
        ranks = np.argsort(p)[:,::-1]

        for idx, item in enumerate(xr):
            out_data.append(item)
            out_pred.append(ranks[idx,:])
            out_emb.append(e[idx,:])
            out_target.append(y[idx])

    # Save
    if is_save==True:
        logger.debug("Saving...")
        if six.PY2:
            with open('%s/data.pkl'%save_path,'w') as f:
                pkl.dump(out_data,f)
            with open('%s/predictions.npy'%save_path,'w') as f:
                np.save(f,np.asarray(out_pred))
            with open('%s/embeddings.npy'%save_path,'w') as f:
                np.save(f,np.asarray(out_emb))
            with open('%s/targets.pkl'%save_path,'w') as f:
                pkl.dump(out_target,f)
        else:
            with open('%s/data.pkl'%save_path,'wb') as f:
                pkl.dump(out_data,f)
            with open('%s/predictions.npy'%save_path,'wb') as f:
                np.save(f,np.asarray(out_pred))
            with open('%s/embeddings.npy'%save_path,'wb') as f:
                np.save(f,np.asarray(out_emb))
            with open('%s/targets.pkl'%save_path,'wb') as f:
                pkl.dump(out_target,f)


if __name__ == '__main__':
    main(sys.argv[1:])
    evaluate.main(sys.argv[3],sys.argv[2])
