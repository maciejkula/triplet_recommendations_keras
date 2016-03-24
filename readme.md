
# Recommendations in Keras using triplet loss
Along the lines of BPR [1]. 

[1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence. AUAI Press, 2009.

This is implemented (more efficiently) in LightFM (https://github.com/lyst/lightfm). See the MovieLens example (https://github.com/lyst/lightfm/blob/master/examples/movielens/example.ipynb) for results comparable to this notebook.

## Set up the architecture
A simple dense layer for both users and items: this is exactly equivalent to latent factor matrix when multiplied by binary user and item indices. There are three inputs: users, positive items, and negative items. In the triplet objective we try to make the positive item rank higher than the negative item for that user.

Because we want just one single embedding for the items, we use shared weights for the positive and negative item inputs (a siamese architecture).

This is all very simple but could be made arbitrarily complex, with more layers, conv layers and so on. I expect we'll be seeing a lot of papers doing just that.



```python
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
```

    Using Theano backend.


## Load and transform data
We're going to load the Movielens 100k dataset and create triplets of (user, known positive item, randomly sampled negative item).

The success metric is AUC: in this case, the probability that a randomly chosen known positive item from the test set is ranked higher for a given user than a ranomly chosen negative item.


```python
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
user_features, positive_item_features, negative_item_features = data.get_dense_triplets(uid,
                                                                                        pid,
                                                                                        nid,
                                                                                        num_users,
                                                                                        num_items)

model = get_graph(num_users, num_items, 256)

# Print the model structure
print(model.summary())

# Sanity check, should be around 0.5
print('AUC before training %s' % metrics.full_auc(model, test))
```

    --------------------------------------------------------------------------------
    Layer (name)                  Output Shape                  Param #             
    --------------------------------------------------------------------------------
    Layer (user_input)            (None, 944)                   0                   
    Layer (positive_item_input)   (None, 1683)                  0                   
    Layer (negative_item_input)   (None, 1683)                  0                   
    Siamese (item_latent)         None                          431104              
    Sequential (user_latent)      (None, 256)                   241920              
    Lambda (triplet_loss)         None                          0                   
    Lambda (triplet_loss)         None                          0                   
    --------------------------------------------------------------------------------
    Total params: 673024
    --------------------------------------------------------------------------------
    None
    AUC before training 0.513835762337


## Run the model
Run for a couple of epochs, checking the AUC after every epoch.


```python
for epoch in range(num_epochs):

    print('Epoch %s' % epoch)

    model.fit({'user_input': user_features,
               'positive_item_input': positive_item_features,
               'negative_item_input': negative_item_features,
               'triplet_loss': np.ones(len(uid))},
              validation_data={'user_input': test_user_features,
                               'positive_item_input': test_positive_item_features,
                               'negative_item_input': test_negative_item_features,
                               'triplet_loss': np.ones(len(uid))},
              batch_size=512,
              nb_epoch=1, 
              verbose=2,
              shuffle=True)

    print('AUC %s' % metrics.full_auc(model, test))
    print('Inversions percentage %s' % count_inversions(model,
                                                        test_user_features,
                                                        test_positive_item_features,
                                                        test_negative_item_features))
```

    Epoch 0
    Train on 49906 samples, validate on 5469 samples
    Epoch 1/1
    2s - loss: -8.5987e-01 - val_loss: -8.4400e-01
    AUC 0.839738215107
    Inversions percentage 0.0
    Epoch 1
    Train on 49906 samples, validate on 5469 samples
    Epoch 1/1
    2s - loss: -8.6342e-01 - val_loss: -8.4455e-01
    AUC 0.837477853849
    Inversions percentage 0.0
    Epoch 2
    Train on 49906 samples, validate on 5469 samples
    Epoch 1/1
    2s - loss: -8.6641e-01 - val_loss: -8.4507e-01
    AUC 0.834460576151
    Inversions percentage 0.0
    Epoch 3
    Train on 49906 samples, validate on 5469 samples
    Epoch 1/1
    2s - loss: -8.6936e-01 - val_loss: -8.4560e-01
    AUC 0.832217327676
    Inversions percentage 0.0
    Epoch 4
    Train on 49906 samples, validate on 5469 samples
    Epoch 1/1
    2s - loss: -8.7201e-01 - val_loss: -8.4597e-01
    AUC 0.82897806716
    Inversions percentage 0.0


The AUC is in the mid-80s. At some point we start overfitting, so it would be a good idea to stop early or add some regularization.


```python

```
