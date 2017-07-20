# Recommendations in Keras using triplet loss

_Note_: a much richer set of neural network recommender models is available as [Spotlight](https://github.com/maciejkula/spotlight).

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

from keras import backend as K
from keras.models import Model
from keras.layers import Embedding, Flatten, Input, merge
from keras.optimizers import Adam

import data
import metrics


def identity_loss(y_true, y_pred):

    return K.mean(y_pred - 0 * y_true)


def bpr_triplet_loss(X):

    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss


def build_model(num_users, num_items, latent_dim):

    positive_item_input = Input((1, ), name='positive_item_input')
    negative_item_input = Input((1, ), name='negative_item_input')

    # Shared embedding layer for positive and negative items
    item_embedding_layer = Embedding(
        num_items, latent_dim, name='item_embedding', input_length=1)

    user_input = Input((1, ), name='user_input')

    positive_item_embedding = Flatten()(item_embedding_layer(
        positive_item_input))
    negative_item_embedding = Flatten()(item_embedding_layer(
        negative_item_input))
    user_embedding = Flatten()(Embedding(
        num_users, latent_dim, name='user_embedding', input_length=1)(
            user_input))

    loss = merge(
        [positive_item_embedding, negative_item_embedding, user_embedding],
        mode=bpr_triplet_loss,
        name='loss',
        output_shape=(1, ))

    model = Model(
        input=[positive_item_input, negative_item_input, user_input],
        output=loss)
    model.compile(loss=identity_loss, optimizer=Adam())

    return model
```

    Using Theano backend.


## Load and transform data
We're going to load the Movielens 100k dataset and create triplets of (user, known positive item, randomly sampled negative item).

The success metric is AUC: in this case, the probability that a randomly chosen known positive item from the test set is ranked higher for a given user than a ranomly chosen negative item.


```python
latent_dim = 100
num_epochs = 10

# Read data
train, test = data.get_movielens_data()
num_users, num_items = train.shape

# Prepare the test triplets
test_uid, test_pid, test_nid = data.get_triplets(test)

model = build_model(num_users, num_items, latent_dim)

# Print the model structure
print(model.summary())

# Sanity check, should be around 0.5
print('AUC before training %s' % metrics.full_auc(model, test))
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    positive_item_input (InputLayer) (None, 1)             0                                            
    ____________________________________________________________________________________________________
    negative_item_input (InputLayer) (None, 1)             0                                            
    ____________________________________________________________________________________________________
    user_input (InputLayer)          (None, 1)             0                                            
    ____________________________________________________________________________________________________
    item_embedding (Embedding)       (None, 1, 100)        168300      positive_item_input[0][0]        
                                                                       negative_item_input[0][0]        
    ____________________________________________________________________________________________________
    user_embedding (Embedding)       (None, 1, 100)        94400       user_input[0][0]                 
    ____________________________________________________________________________________________________
    flatten_7 (Flatten)              (None, 100)           0           item_embedding[0][0]             
    ____________________________________________________________________________________________________
    flatten_8 (Flatten)              (None, 100)           0           item_embedding[1][0]             
    ____________________________________________________________________________________________________
    flatten_9 (Flatten)              (None, 100)           0           user_embedding[0][0]             
    ____________________________________________________________________________________________________
    loss (Merge)                     (None, 1)             0           flatten_7[0][0]                  
                                                                       flatten_8[0][0]                  
                                                                       flatten_9[0][0]                  
    ====================================================================================================
    Total params: 262700
    ____________________________________________________________________________________________________
    None
    AUC before training 0.50247407966


## Run the model
Run for a couple of epochs, checking the AUC after every epoch.


```python
for epoch in range(num_epochs):

    print('Epoch %s' % epoch)

    # Sample triplets from the training data
    uid, pid, nid = data.get_triplets(train)

    X = {
        'user_input': uid,
        'positive_item_input': pid,
        'negative_item_input': nid
    }

    model.fit(X,
              np.ones(len(uid)),
              batch_size=64,
              nb_epoch=1,
              verbose=0,
              shuffle=True)

    print('AUC %s' % metrics.full_auc(model, test))
```

    Epoch 0
    AUC 0.905896400776
    Epoch 1
    AUC 0.908241780938
    Epoch 2
    AUC 0.909650205748
    Epoch 3
    AUC 0.910820451523
    Epoch 4
    AUC 0.912184845152
    Epoch 5
    AUC 0.912632057958
    Epoch 6
    AUC 0.91326604222
    Epoch 7
    AUC 0.913786881853
    Epoch 8
    AUC 0.914638438854
    Epoch 9
    AUC 0.915375014253


The AUC is in the low-90s. At some point we start overfitting, so it would be a good idea to stop early or add some regularization.
