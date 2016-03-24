"""
Triplet loss network example for recommenders
"""


from __future__ import print_function

import numpy as np

import theano

import keras
from keras import backend as K
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Lambda
from keras.optimizers import Adagrad, Adam


import data
import metrics


def identity_loss(y_true, y_pred):

    return K.mean(y_pred - 0 * y_true)


def bpr_triplet_loss(X):

    user_latent, item_latent = X.values()
    positive_item_latent, negative_item_latent = item_latent.values()

    # BPR loss
    loss = - 1.0 / (1.0 + K.exp(-(K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True)
                                - K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))))

    return loss


def margin_triplet_loss(X):

    user_latent, item_latent = X.values()
    positive_item_latent, negative_item_latent = item_latent.values()

    # Hinge loss: max(0, user * negative_item_latent + 1 - user * positive_item_latent)
    loss = K.maximum(1.0
                     + K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True)
                     - K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True),
                     0.0)

    return loss


def get_item_subgraph(input_shape, latent_dim):
    # Could take item metadata here, do convolutional layers etc.

    model = Sequential()
    model.add(Dense(latent_dim, input_shape=input_shape))

    return model


def get_user_subgraph(input_shape, latent_dim):
    # Could do all sorts of fun stuff here that takes
    # user metadata in.

    model = Sequential()
    model.add(Dense(latent_dim, input_shape=input_shape))

    return model


def get_graph(num_users, num_items, latent_dim):

    batch_input_shape = (1,)

    model = Graph()

    # Add inputs
    model.add_input('user_input', input_shape=(num_users,), batch_input_shape=batch_input_shape)
    model.add_input('positive_item_input', input_shape=(num_items,), batch_input_shape=batch_input_shape)
    model.add_input('negative_item_input', input_shape=(num_items,), batch_input_shape=batch_input_shape)

    # Add shared-weight item subgraph
    model.add_shared_node(get_item_subgraph((num_items,), latent_dim),
                          name='item_latent',
                          inputs=['positive_item_input',
                                  'negative_item_input'],
                          merge_mode='join')
    # Add user embedding
    model.add_node(get_user_subgraph((num_users,), latent_dim),
                   name='user_latent',
                   input='user_input')

    # Compute loss
    model.add_node(Lambda(bpr_triplet_loss),
                   name='triplet_loss',
                   inputs=['user_latent', 'item_latent'],
                   merge_mode='join')

    # Add output
    model.add_output(name='triplet_loss', input='triplet_loss')

    # Compile using a dummy loss to fit within the Keras paradigm
    model.compile(loss={'triplet_loss': identity_loss}, optimizer=Adam())#Adagrad(lr=0.1, epsilon=1e-06))

    return model


def count_inversions(model, user_features, posititve_item_features, negative_item_features):

    loss = model.predict({'user_input': user_features,
                          'positive_item_input': posititve_item_features,
                          'negative_item_input': negative_item_features})['triplet_loss']

    return (loss > 0).mean()


if __name__ == '__main__':

    num_epochs = 5

    # Read data
    train, test = data.get_movielens_data()
    num_users, num_items = train.shape

    # Prepare the test triplets
    test_uid, test_pid, test_nid = data.get_triplets(test)
    test_user_features, test_positive_item_features, test_negative_item_features = data.get_dense_triplets(test_uid,
                                                                                                           test_pid,
                                                                                                           test_nid,
                                                                                                           num_users,
                                                                                                           num_items)

    # Sample triplets from the training data
    uid, pid, nid = data.get_triplets(train)
    user_features, positive_item_features, negative_item_features = data.get_dense_triplets(uid, pid, nid, num_users, num_items)

    model = get_graph(num_users, num_items, 256)

    # Print the model structure
    print(model.summary())

    # Sanity check, should be around 0.5
    print('AUC before training %s' % metrics.full_auc(model, test))

    for epoch in range(num_epochs):

        print('Epoch %s' % epoch)

        model.fit({'user_input': user_features,
                   'positive_item_input': positive_item_features,
                   'negative_item_input': negative_item_features, 'triplet_loss': np.ones(len(uid))},
                  validation_data={'user_input': test_user_features,
                                   'positive_item_input': test_positive_item_features,
                                   'negative_item_input': test_negative_item_features, 'triplet_loss': np.ones(len(uid))},
                  batch_size=512,
                  nb_epoch=1, 
                  verbose=2,
                  shuffle=True)

        print('AUC %s' % metrics.full_auc(model, test))
        print('Inversions percentage %s' % count_inversions(model,
                                                            test_user_features,
                                                            test_positive_item_features,
                                                            test_negative_item_features))

