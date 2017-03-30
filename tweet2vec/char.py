'''
Tweet2Vec classifier trainer
'''

# modules
from tweet2vec import batch_char as batch
from tweet2vec.t2v import tweet2vec, init_params, load_params_shared
from tweet2vec.settings_char import NUM_EPOCHS, N_BATCH, MAX_LENGTH, SCALE, WDIM, MAX_CLASSES, LEARNING_RATE, DISPF, SAVEF, REGULARIZATION, RELOAD_MODEL, MOMENTUM, SCHEDULE
from tweet2vec.evaluate import precision
# data object
from theano.tensor import TensorVariable
from lasagne.layers import DenseLayer
# typing
from typing import List, Tuple, Any
# core
import numpy as np
# theano
import theano
import theano.tensor as T
# logger
from tweet2vec.logger import logger
# else
import lasagne
import random
import sys
import six
import time
import io
import shutil
from collections import OrderedDict
if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl


T1 = 0.01
T2 = 0.0001


def schedule(lr, mu):
    logger.debug("Updating Schedule...")
    lr = max(1e-5,lr/2)
    return lr, mu


def tnorm(tens):
    '''
    Tensor Norm
    '''
    return T.sqrt(T.sum(T.sqr(tens),axis=1))


def classify(tweet, t_mask, params, n_classes, n_chars):
    """* What you can do

    """
    # type: (TensorVariable,TensorVariable,OrderedDict,int)->Tuple[TensorVariable,DenseLayer,TensorVariable]
    # tweet embedding
    emb_layer = tweet2vec(tweet, t_mask, params, n_chars)
    # Dense layer for classes
    l_dense = lasagne.layers.DenseLayer(emb_layer, n_classes, W=params['W_cl'], b=params['b_cl'], nonlinearity=lasagne.nonlinearities.softmax)

    return lasagne.layers.get_output(l_dense), l_dense, lasagne.layers.get_output(emb_layer)



def main(train_path, val_path, save_path, num_epochs=NUM_EPOCHS):
    """"""
    # type: (str,str,str,int)->None
    global T1

    # save settings
    shutil.copyfile('settings_char.py','%s/settings_char.txt'%save_path)

    logger.debug("Preparing Data...")
    # Training data
    Xt = []
    yt = []
    with io.open(train_path,'r',encoding='utf-8') as f:
        for line in f:
            (yc, Xc) = line.rstrip('\n').split('\t')
            Xt.append(Xc[:MAX_LENGTH])
            yt.append(yc)
    # Validation data
    Xv = []
    yv = []
    with io.open(val_path,'r',encoding='utf-8') as f:
        for line in f:
            (yc, Xc) = line.rstrip('\n').split('\t')
            Xv.append(Xc[:MAX_LENGTH])
            yv.append(yc.split(','))

    logger.debug("Building Model...")
    if not RELOAD_MODEL:
        # Build dictionaries from training data
        chardict, charcount = batch.build_dictionary(Xt)
        n_char = len(chardict.keys()) + 1
        batch.save_dictionary(chardict,charcount,'%s/dict.pkl' % save_path)
        # params
        params = init_params(n_chars=n_char)
        
        labeldict, labelcount = batch.build_label_dictionary(yt)
        batch.save_dictionary(labeldict, labelcount, '%s/label_dict.pkl' % save_path)

        n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)

        # classification params
        params['W_cl'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(WDIM,n_classes)).astype('float32'), name='W_cl')
        params['b_cl'] = theano.shared(np.zeros((n_classes)).astype('float32'), name='b_cl')

    else:
        logger.debug("Loading model params...")
        params = load_params_shared('%s/model.npz' % save_path)

        logger.debug("Loading dictionaries...")
        with open('%s/dict.pkl' % save_path, 'rb') as f:
            chardict = pkl.load(f)
        with open('%s/label_dict.pkl' % save_path, 'rb') as f:
            labeldict = pkl.load(f)
        n_char = len(chardict.keys()) + 1
        n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)

    # iterators
    train_iter = batch.BatchTweets(Xt, yt, labeldict, batch_size=N_BATCH, max_classes=MAX_CLASSES)
    val_iter = batch.BatchTweets(Xv, yv, labeldict, batch_size=N_BATCH, max_classes=MAX_CLASSES, test=True)

    logger.debug("Building network...")
    # Tweet variables
    tweet = T.itensor3()  # type: TensorVariable
    targets = T.ivector()  # type: TensorVariable
    # masks
    t_mask = T.fmatrix()  # type: TensorVariable

    # network for prediction
    predictions, net, emb = classify(tweet, t_mask, params, n_classes, n_char)

    # batch loss
    loss = lasagne.objectives.categorical_crossentropy(predictions, targets)
    cost = T.cast(T.mean(loss, dtype='float32') + REGULARIZATION*lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2), 'float32')
    cost_only = T.mean(loss, dtype='float32')
    reg_only = REGULARIZATION*lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2)

    # params and updates
    logger.debug("Computing updates...")
    lr = LEARNING_RATE
    mu = MOMENTUM
    updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(net), lr, momentum=mu)

    # Theano function
    logger.debug("Compiling theano functions...")
    inps = [tweet,t_mask,targets]  # type: List[TensorVariable,TensorVariable,TensorVariable]
    predict = theano.function([tweet,t_mask],predictions)
    cost_val = theano.function(inps,[cost_only,emb])
    # it converts all dtype into float32
    updates = [(key, T.cast(updates[key], 'float32')) for key in list(updates.keys())]
    train = theano.function(inputs=inps, outputs=cost, updates=updates)
    reg_val = theano.function([],reg_only)

    # Training
    logger.debug("Training...")
    uidx = 0
    maxp = 0.
    start = time.time()
    valcosts = []
    try:
        for epoch in range(num_epochs):
            n_samples = 0
            train_cost = 0.0
            logger.debug("Epoch {}".format(epoch))

            # learning schedule
            if len(valcosts) > 1 and SCHEDULE:
                change = (valcosts[-1]-valcosts[-2])/abs(valcosts[-2])
                if change < T1:
                    lr, mu = schedule(lr, mu)
                    updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(net), lr, momentum=mu)
                    updates = [(key, T.cast(updates[key], 'float32')) for key in list(updates.keys())]
                    train = theano.function(inputs=inps,outputs=cost,updates=updates)
                    T1 = T1/2

            # stopping criterion
            if len(valcosts) > 6:
                deltas = []
                for i in range(5):
                    deltas.append((valcosts[-i-1]-valcosts[-i-2])/abs(valcosts[-i-2]))
                if sum(deltas)/len(deltas) < T2:
                    break

            ud_start = time.time()
            for xr,y in train_iter:
                n_samples +=len(xr)
                uidx += 1
                x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
                if x is None:
                    logger.debug("Minibatch with zero samples under maxlength.")
                    uidx -= 1
                    continue

                curr_cost = train(x,x_m,y)
                train_cost += curr_cost*len(xr)
                ud = time.time() - ud_start

                if np.isnan(curr_cost) or np.isinf(curr_cost):
                    logger.debug("Nan detected.")
                    return

                if np.mod(uidx, DISPF) == 0:
                    logger.debug("Epoch {} Update {} Cost {} Time {}".format(epoch,uidx,curr_cost,ud))


                if np.mod(uidx,SAVEF) == 0:
                    logger.debug("Saving...")
                    saveparams = OrderedDict()
                    for kk,vv in params.iteritems():
                        saveparams[kk] = vv.get_value()
                        np.savez('%s/model.npz' % save_path,**saveparams)
                        print("Done.")

            logger.debug("Testing on Validation set...")
            preds = []
            targs = []
            for xr,y in val_iter:
                x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
                if x is None:
                    logger.debug("Validation: Minibatch with zero samples under maxlength.")
                    continue

                vp = predict(x,x_m)
                ranks = np.argsort(vp)[:,::-1]
                for idx,item in enumerate(xr):
                    preds.append(ranks[idx,:])
                    targs.append(y[idx])

            validation_cost = precision(np.asarray(preds),targs,1)
            regularization_cost = reg_val()

            if validation_cost > maxp:
                maxp = validation_cost
                saveparams = OrderedDict()
                if six.PY2:
                    for kk,vv in params.iteritems():
                        saveparams[kk] = vv.get_value()
                else:
                    for kk,vv in params.items():
                        saveparams[kk] = vv.get_value()

                np.savez('%s/best_model.npz' % (save_path),**saveparams)

            logger.debug("Epoch {} Training Cost {} Validation Precision {} Regularization Cost {} Max Precision {}".format(epoch, train_cost/n_samples, validation_cost, regularization_cost, maxp))
            logger.debug("Seen {} samples.".format(n_samples))
            valcosts.append(validation_cost)

            logger.debug("Saving...")
            saveparams = OrderedDict()
            if six.PY2:
                for kk,vv in params.iteritems():
                    saveparams[kk] = vv.get_value()
            else:
                for kk,vv in params.items():
                    saveparams[kk] = vv.get_value()
            np.savez('%s/model_%d.npz' % (save_path,epoch),**saveparams)
            logger.debug("Done.")

    except KeyboardInterrupt:
        pass
    logger.debug("Total training time = {}".format(time.time()-start))

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])
