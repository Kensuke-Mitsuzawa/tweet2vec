#! -*- coding: utf-8 -*-
# modules
from tweet2vec import batch_char as batch
from tweet2vec.t2v import tweet2vec, init_params, load_params_shared, load_params
from tweet2vec.settings_char import NUM_EPOCHS, N_BATCH, SCALE, WDIM, MAX_CLASSES, LEARNING_RATE, DISPF, SAVEF, REGULARIZATION, RELOAD_MODEL, MOMENTUM, SCHEDULE
from tweet2vec.evaluate import precision
# data object
from theano.tensor import TensorVariable
from lasagne.layers import DenseLayer
from numpy import ndarray
# typing
from typing import List, Tuple, Any, Union
# core
import numpy as np
# theano
import theano
import theano.tensor as T
# logger
from tweet2vec.logger import logger
# else
import glob
import lasagne
import random
import sys
import six
import time
import os
import io
import shutil
import tempfile
from collections import OrderedDict, namedtuple
if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl


T1 = 0.01
T2 = 0.0001


class PostRecordObject(object):
    """This class is for generic input"""
    __slots__ = ('post_id', 'post_text', 'post_label')
    def __init__(self,
                 post_id,
                 post_text,
                 post_label):
        """"""
        # type: (int,str,str)->None
        self.post_id = post_id
        self.post_text = post_text
        self.post_label = post_label


class PredictionRecordObject(object):
    __slots__ = ('post_id', 'post_text', 'embedding_vector', 'prediction_label')
    def __init__(self,
                 post_text,
                 embedding_vector,
                 prediction_label,
                 post_id=None):
        self.post_id = post_id
        self.post_text = post_text
        self.embedding_vector = embedding_vector
        self.prediction_label = prediction_label



class InputDataset(object):
    """This is Interface class to handle input text data"""
    def __init__(self, text, label=None, text_id=None):
        """"""
        # type: (Union[List[str],OrderedDict],Union[List[str],OrderedDict],Union[List[str],OrderedDict])->None
        self.text = text
        self.label = label
        self.text_id = text_id

    @classmethod
    def load_from_generic_input(cls, seq_post):
        """* What you can do
        """
        # type: (List[PostRecordObject])
        from tweet2vec.settings_char import MAX_LENGTH

        seq_text = [None] * len(seq_post)
        seq_label = [None] * len(seq_post)
        seq_text_id = [None] * len(seq_post)
        for i, record_obj in enumerate(seq_post):
            seq_label[i] = record_obj.post_label
            seq_text_id[i] = record_obj.post_id
            if len(record_obj.post_text) > MAX_LENGTH:
                logger.warning("""
                Input text is longer than MAX_LENGTH. It cuts off automatically. You might consider to set bigger value on settings_char.MAX_LENGTH.
                Input={}""".format(record_obj.post_text))
                seq_text[i] = record_obj.post_text[:(MAX_LENGTH-1)]
            else:
                seq_text[i] = record_obj.post_text

        return InputDataset(seq_text, seq_label, seq_text_id)

    @classmethod
    def load_table_data(cls, path_to_data, separator='\t'):
        """* What you can do
        - It loads data from file on the local-disk

        * File Format
        - Tab separated file, which has label\tsentence
            - For example
            - summer\tgood morning san diego!!!
        """
        # type: (str,str)->InputDataset
        from tweet2vec.settings_char import MAX_LENGTH
        logger.debug("Preparing Data...")
        Xt = []
        yt = []
        with io.open(path_to_data, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    (yc, Xc) = line.rstrip('\n').split(separator)
                except:
                    logger.warning('skip the line={}'.format(line))
                else:
                    Xt.append(Xc[:MAX_LENGTH])
                    yt.append(yc)
        return InputDataset(Xt, yt)


class ModelObject(object):
    """This is data object class to handle trained-model"""
    def __init__(self,
                 best_model,
                 seq_epoch_model,
                 chardict,
                 labeldict,
                 charcount=None,
                 labelcount=None
                 ):
        """* Parameters"""
        # type: (OrderedDict,List[OrderedDict],OrderedDict,OrderedDict,OrderedDict,OrderedDict)->None
        self.best_model = best_model
        self.seq_epoch_model = seq_epoch_model
        self.chardict = chardict
        self.labeldict = labeldict
        self.charcount = charcount
        self.labelcount = labelcount

    def save_model(self, save_dir):
        """* What you can do
        - It saves model files on the disk
        """
        # type: (str)->None
        if not os.path.exists(save_dir):
            raise Exception('There is no directory at {}'.format(save_dir))

        batch.save_dictionary(self.chardict, self.charcount, '%s/dict.pkl' % save_dir)
        batch.save_dictionary(self.labeldict, self.labelcount, '%s/label_dict.pkl' % save_dir)
        np.savez('%s/best_model.npz' % (save_dir), **self.best_model)
        for i, model_params in enumerate(self.seq_epoch_model):
            np.savez('%s/model_%d.npz' % (save_dir, i), **model_params)

    @classmethod
    def load_model(self, model_dir):
        """* What you can do
        - It loads model/dictionaries from given directory
        """
        if not os.path.exists('%s/dict.pkl' % model_dir):
            raise Exception('There is no trained model %s/dict.pkl', model_dir)
        if not os.path.exists('%s/label_dict.pkl' % model_dir):
            raise Exception('There is no trained model %s/label_dict.pkl', model_dir)

        logger.debug("Loading models...")
        best_params = load_params('%s/best_model.npz' % model_dir)
        seq_epoch_model = [load_params('%s' % (model_file))
                           for model_file in glob.glob(os.path.join(model_dir, 'model_*.npz'))]

        logger.debug("Loading dictionaries...")
        with open('%s/dict.pkl' % model_dir, 'rb') as f:
            chardict = pkl.load(f)
        with open('%s/label_dict.pkl' % model_dir, 'rb') as f:
            labeldict = pkl.load(f)

        return ModelObject(
            best_model=best_params,
            seq_epoch_model=seq_epoch_model,
            chardict=chardict,
            labeldict=labeldict,
            charcount=None,
            labelcount=None)


class Twee2vecInterface(object):
    def __init__(self,
                 working_dir=tempfile.mkdtemp()):
        pass

    def invert(self, d):
        out = {}
        if six.PY2:
            for k,v in d.iteritems():
                out[v] = k
        else:
            for k,v in d.items():
                out[v] = k

        return out

    def schedule(self, lr, mu):
        logger.debug("Updating Schedule...")
        lr = max(1e-5, lr / 2)
        return lr, mu

    def tnorm(self, tens):
        '''
        Tensor Norm
        '''
        return T.sqrt(T.sum(T.sqr(tens), axis=1))

    def classify(self, tweet, t_mask, params, n_classes, n_chars, is_test):
        """* What you can do
        - It returns embedded tensor-variable

        * Parameter
        - is_test: it does not return class information if True. This uses for prediction step.
        """
        # type: (TensorVariable,TensorVariable,OrderedDict,int,bool)->Union[Tuple[TensorVariable,DenseLayer,TensorVariable],Tuple[TensorVariable,TensorVariable]]
        # tweet embedding
        emb_layer = tweet2vec(tweet, t_mask, params, n_chars)
        # Dense layer for classes
        l_dense = lasagne.layers.DenseLayer(emb_layer, n_classes, W=params['W_cl'], b=params['b_cl'],
                                            nonlinearity=lasagne.nonlinearities.softmax)
        if is_test:
            return lasagne.layers.get_output(l_dense), lasagne.layers.get_output(emb_layer)
        else:
            return lasagne.layers.get_output(l_dense), l_dense, lasagne.layers.get_output(emb_layer)

    def construct_char_dict(self, input_dataset):
        """* What you can do
        - It constructs character feature dictionary
        """
        # type: (InputDataset)->Tuple[OrderedDict,OrderedDict]
        # Build dictionaries from training data
        chardict, charcount = batch.build_dictionary(text=input_dataset.text)

        return chardict, charcount

    def construct_label_dict(self, input_dataset):
        """* What you can do
        - It constructs label dictionary
        """
        # type: (OrderedDict,InputDataset)->Tuple[OrderedDict,OrderedDict,int]
        labeldict, labelcount = batch.build_label_dictionary(targets=input_dataset.label)

        n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)

        return labeldict, labelcount, n_classes

    def init_params(self, n_char, n_classes):
        """* What you can do
        - It make initialization of parameter object
        """
        # type: (int,int)->OrderedDict
        # params
        params = init_params(n_chars=n_char)
        # classification params
        params['W_cl'] = theano.shared(
            np.random.normal(loc=0., scale=SCALE, size=(WDIM, n_classes)).astype('float32'), name='W_cl')
        params['b_cl'] = theano.shared(np.zeros((n_classes)).astype('float32'), name='b_cl')

        return params

    def train(self,
              training_dataset,
              validation_dataset,
              save_dir,
              is_use_trained_model=False,
              num_epochs=NUM_EPOCHS,
              model_object=None):
        """* What you can do
        - It trains model from given data.

        * Parameters
        - training_dataset
        - validation_dataset
        """
        # type: (InputDataset,InputDataset,str,bool,int,ModelObject)->ModelObject
        global T1

        if not os.path.exists(save_dir):
            raise Exception('There is no directory at {}'.format(save_dir))
        logger.debug(msg='save_path = {}'.format(save_dir))

        ## Makes character-feature dictionary & label-dictionary ##
        logger.debug("Building Model...")
        if not is_use_trained_model:
            chardict, charcount = self.construct_char_dict(training_dataset)
            labeldict, labelcount, n_classes = self.construct_label_dict(training_dataset)
            n_char = len(chardict.keys()) + 1
            params = self.init_params(n_char, n_classes)
        else:
            if model_object is None:
                raise Exception('You must given path to loaded model object to "model_object"')
            labeldict = model_object.labeldict
            chardict = model_object.chardict
            params = model_object.best_model
            charcount = None
            labelcount = None

            if six.PY2:
                n_char = len(chardict.keys()) + 1
                n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)
            else:
                n_char = len(list(chardict.keys())) + 1
                n_classes = min(len(list(labeldict.keys())) + 1, MAX_CLASSES)

        # iterators
        train_iter = batch.BatchTweets(data=training_dataset.text,
                                       targets=training_dataset.label,
                                       labeldict=labeldict,
                                       batch_size=N_BATCH,
                                       max_classes=MAX_CLASSES)
        val_iter = batch.BatchTweets(data=validation_dataset.text,
                                     targets=validation_dataset.label,
                                     labeldict=labeldict,
                                     batch_size=N_BATCH,
                                     max_classes=MAX_CLASSES,
                                     test=True)

        logger.debug("Building network...")
        # Tweet variables
        tweet = T.itensor3()  # type: TensorVariable
        targets = T.ivector()  # type: TensorVariable
        # masks
        t_mask = T.fmatrix()  # type: TensorVariable

        # network for prediction
        predictions, net, emb = self.classify(tweet, t_mask, params, n_classes, n_char, is_test=False)

        # batch loss
        loss = lasagne.objectives.categorical_crossentropy(predictions, targets)
        cost = T.cast(
            T.mean(loss, dtype='float32') \
            + REGULARIZATION * lasagne.regularization.regularize_network_params(net,lasagne.regularization.l2),
            'float32')
        cost_only = T.mean(loss, dtype='float32')
        reg_only = REGULARIZATION * lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2)

        # params and updates
        logger.debug("Computing updates...")
        lr = LEARNING_RATE
        mu = MOMENTUM
        updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(net), lr, momentum=mu)

        # Theano function
        logger.debug("Compiling theano functions...")
        inps = [tweet, t_mask, targets]  # type: List[TensorVariable,TensorVariable,TensorVariable]
        predict = theano.function([tweet, t_mask], predictions)
        cost_val = theano.function(inps, [cost_only, emb])
        # it converts all dtype into float32
        updates = [(key, T.cast(updates[key], 'float32')) for key in list(updates.keys())]
        train = theano.function(inputs=inps, outputs=cost, updates=updates)
        reg_val = theano.function([], reg_only)

        # Training
        logger.info("Training...")
        uidx = 0
        maxp = 0.0
        start = time.time()
        valcosts = []

        best_model = OrderedDict()
        seq_epoch_model = []  # type: List[OrderedDict]

        try:
            for epoch in range(num_epochs):
                n_samples = 0
                train_cost = 0.0
                logger.info("Epoch {}".format(epoch))

                # learning schedule
                if len(valcosts) > 1 and SCHEDULE:
                    change = (valcosts[-1] - valcosts[-2]) / abs(valcosts[-2])
                    if change < T1:
                        lr, mu = self.schedule(lr, mu)
                        updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(net), lr,
                                                                    momentum=mu)
                        updates = [(key, T.cast(updates[key], 'float32')) for key in list(updates.keys())]
                        train = theano.function(inputs=inps, outputs=cost, updates=updates)
                        T1 = T1 / 2

                # stopping criterion
                if len(valcosts) > 6:
                    deltas = []
                    for i in range(5):
                        deltas.append((valcosts[-i - 1] - valcosts[-i - 2]) / abs(valcosts[-i - 2]))
                    if sum(deltas) / len(deltas) < T2:
                        break

                ud_start = time.time()
                for xr, y in train_iter:
                    n_samples += len(xr)
                    uidx += 1
                    x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
                    if x is None:
                        logger.debug("Minibatch with zero samples under maxlength.")
                        uidx -= 1
                        continue

                    curr_cost = train(x, x_m, y)
                    train_cost += curr_cost * len(xr)
                    ud = time.time() - ud_start

                    if np.isnan(curr_cost) or np.isinf(curr_cost):
                        logger.debug("Nan detected.")
                        return

                    if np.mod(uidx, DISPF) == 0:
                        logger.debug("Epoch {} Update {} Cost {} Time {}".format(epoch, uidx, curr_cost, ud))

                    if np.mod(uidx, SAVEF) == 0:
                        logger.debug("Saving...")
                        saveparams = OrderedDict()
                        if six.PY2:
                            for kk, vv in params.iteritems():
                                saveparams[kk] = vv.get_value()
                                np.savez('%s/model.npz' % save_dir, **saveparams)
                                logger.debug("Done.")
                        else:
                            for kk, vv in params.items():
                                saveparams[kk] = vv.get_value()
                                np.savez('%s/model.npz' % save_dir, **saveparams)
                                logger.debug("Done.")

                logger.debug("Testing on Validation set...")
                preds = []
                targs = []
                for xr, y in val_iter:
                    x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
                    if x is None:
                        logger.debug("Validation: Minibatch with zero samples under maxlength.")
                        continue

                    vp = predict(x, x_m)
                    ranks = np.argsort(vp)[:, ::-1]
                    for idx, item in enumerate(xr):
                        preds.append(ranks[idx, :])
                        targs.append(y[idx])

                validation_cost = precision(np.asarray(preds), targs, 1)
                regularization_cost = reg_val()

                if validation_cost > maxp:
                    maxp = validation_cost
                    saveparams = OrderedDict()
                    if six.PY2:
                        for kk, vv in params.iteritems():
                            saveparams[kk] = vv.get_value()
                    else:
                        for kk, vv in params.items():
                            saveparams[kk] = vv.get_value()
                    best_model = saveparams
                    #np.savez('%s/best_model.npz' % (save_dir), **saveparams)

                logger.info(
                    "Epoch {} Training Cost {} Validation Precision {} Regularization Cost {} Max Precision {}".format(
                        epoch, train_cost / n_samples, validation_cost, regularization_cost, maxp))
                logger.debug("Seen {} samples.".format(n_samples))
                valcosts.append(validation_cost)

                saveparams = OrderedDict()
                if six.PY2:
                    for kk, vv in params.iteritems():
                        saveparams[kk] = vv.get_value()
                else:
                    for kk, vv in params.items():
                        saveparams[kk] = vv.get_value()
                seq_epoch_model.append(saveparams)

        except KeyboardInterrupt:
            pass

        ### When validation score is always 0.0 ###
        if len(best_model) == 0:
            logger.warning(msg='Validation cost is always 0.0. It might be better to set different data.')
            best_model = seq_epoch_model[-1]

        logger.info(msg='End training!')
        logger.info(msg="Total training time = {}".format(time.time() - start))

        model_object = ModelObject(best_model=best_model,
                                   seq_epoch_model=seq_epoch_model,
                                   chardict=chardict,
                                   labeldict=labeldict,
                                   charcount=charcount,
                                   labelcount=labelcount)
        model_object.save_model(save_dir=save_dir)

        return model_object


    def predict(self, test_data, model_object):
        """* What you can do
        """
        # type: (InputDataset,ModelObject)->List[PredictionRecordObject]

        # Model
        logger.debug("Loading model params...")

        params = model_object.best_model
        chardict = model_object.chardict
        n_char = len(model_object.chardict.keys()) + 1
        n_classes = min(len(model_object.labeldict.keys()) + 1, MAX_CLASSES)
        inverse_labeldict = self.invert(model_object.labeldict)

        logger.debug("Building network...")
        # Tweet variables
        tweet = T.itensor3()
        t_mask = T.fmatrix()

        # network for prediction
        predictions, embeddings = self.classify(tweet, t_mask, params, n_classes, n_char, is_test=True)

        # Theano function
        logger.debug("Compiling theano functions...")
        predict = theano.function([tweet, t_mask], predictions)
        encode = theano.function([tweet, t_mask], embeddings)

        # Test
        logger.debug("Encoding...")
        out_pred = []
        out_emb = []
        if six.PY2:
            numbatches = len(test_data.text) / N_BATCH + 1
        else:
            numbatches = int(len(test_data.text) / N_BATCH) + 1

        for i in range(numbatches):
            xr = test_data.text[N_BATCH * i:N_BATCH * (i + 1)]
            x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
            p = predict(x, x_m)
            e = encode(x, x_m)
            ranks = np.argsort(p)[:, ::-1]

            for idx, item in enumerate(xr):
                out_pred.append( tuple([inverse_labeldict[r] if r in inverse_labeldict else 'UNK' for r in ranks[idx, :5]]) )
                out_emb.append(e[idx, :])

        seq_prediction_result = [None] * len(test_data.text)
        if test_data.text_id is None:
            seq_text_id = [None] * len(test_data.text)
        else:
            seq_text_id = test_data.text_id
        for i, text in enumerate(test_data.text):
            seq_prediction_result[i] = PredictionRecordObject(post_text=text,
                                                              embedding_vector=out_emb[i],
                                                              prediction_label=out_pred[i],
                                                              post_id=seq_text_id[i])

        return seq_prediction_result

    def evaluate(self):
        """* What you can do
        """
        pass

