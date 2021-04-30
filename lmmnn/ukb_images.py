import os
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.keras.layers import (Concatenate, Conv2D, Dense, Dropout,
                                     Embedding, Flatten, Input, MaxPool2D,
                                     Reshape, GlobalAveragePooling2D)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from lmmnn.callbacks import EarlyStoppingWithSigmasConvergence, PrintSigmas
from lmmnn.layers import NLL

logger = logging.getLogger('UKB.logger')

IMG_WIDTH = 299

def sample_split(seed, n_subjects, train_frac=0.72, valid_n = 1600):
    np.random.seed(seed)
    # n_subjects = np.max(images_df['subject_id2'].unique()) + 1
    train_samp_subj = np.random.choice(n_subjects, int(train_frac * n_subjects), replace=False)
    valid_samp_subj = np.random.choice(np.delete(np.arange(n_subjects), train_samp_subj), valid_n, replace=False)
    test_samp_subj = np.delete(np.arange(n_subjects), np.concatenate([train_samp_subj, valid_samp_subj]))
    return train_samp_subj, valid_samp_subj, test_samp_subj

def cnn_ignore():
    input_layer = Input((IMG_WIDTH, IMG_WIDTH, 3))
    x = Conv2D(32, (5, 5), activation='relu')(input_layer)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(32, (5, 5), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    output = Dense(1)(x)
    return Model(inputs=[input_layer], outputs=output)

def cnn_ignore_inception():
    base = InceptionV3(weights='imagenet', include_top=False, input_shape = (IMG_WIDTH, IMG_WIDTH, 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(1)(x)
    model = Model(inputs = base.input, outputs = output)
    train_top = 55
    for layer in model.layers[:-train_top]:
        layer.trainable = False
    for layer in model.layers[-train_top:]:
        layer.trainable = True
    return model

def cnn_lmmnn():
    input_layer = Input((IMG_WIDTH, IMG_WIDTH, 3))
    y_true_input = Input(shape=(1, ),)
    Z_input = Input(shape=(1, ), dtype=tf.int64)
    x = Conv2D(32, (5, 5), activation='relu')(input_layer)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(32, (5, 5), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    y_pred_output = Dense(1)(x)
    nll = NLL(1.0, 1.0)(y_true_input, y_pred_output, Z_input)
    return Model(inputs=[input_layer, y_true_input, Z_input], outputs=nll)

def cnn_lmmnn_inception():
    y_true_input = Input(shape=(1, ),)
    Z_input = Input(shape=(1, ), dtype=tf.int64)
    base = InceptionV3(weights='imagenet', include_top=False, input_shape = (IMG_WIDTH, IMG_WIDTH, 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    y_pred_output = Dense(1)(x)
    nll = NLL(1.0, 1.0)(y_true_input, y_pred_output, Z_input)
    model = Model(inputs=[base.input, y_true_input, Z_input], outputs=nll)
    train_top = 58
    for layer in model.layers[:-train_top]:
        layer.trainable = False
    for layer in model.layers[-train_top:]:
        layer.trainable = True
    return model

def cnn_embedding(n_cats, embed_dim):
    input_layer = Input((IMG_WIDTH, IMG_WIDTH, 3))
    Z_input = Input(shape=(1,))
    embed = Embedding(n_cats, embed_dim, input_length = 1)(Z_input)
    embed = Reshape(target_shape = (embed_dim,))(embed)
    x = Conv2D(32, (5, 5), activation='relu')(input_layer)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(32, (5, 5), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    concat = Concatenate()([x, embed])
    output = Dense(1)(concat)
    return Model(inputs=[input_layer, Z_input], outputs=output)

def cnn_embedding_inception(n_cats, embed_dim):
    base = InceptionV3(weights='imagenet', include_top=False, input_shape = (IMG_WIDTH, IMG_WIDTH, 3))
    Z_input = Input(shape=(1,))
    embed = Embedding(n_cats, embed_dim, input_length = 1)(Z_input)
    embed = Reshape(target_shape = (embed_dim,))(embed)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    concat = Concatenate()([x, embed])
    output = Dense(1)(concat)
    model = Model(inputs = [base.input, Z_input], outputs = output)
    train_top = 55
    print(model.layers)
    for layer in model.layers[:-train_top]:
        layer.trainable = False
    for layer in model.layers[-train_top:]:
        layer.trainable = True
    return model

def calc_b_hat(Z_train, y_train, y_pred_tr, n_cats, sig2e, sig2b):
    b_hat = []
    for i in range(n_cats):
        i_vec = Z_train == i
        n_i = i_vec.sum()
        if n_i > 0:
            y_bar_i = y_train[i_vec].mean()
            y_pred_i = y_pred_tr[i_vec].mean()
            # BP(b_i) = (n_i * sig2b / (sig2a + n_i * sig2b)) * (y_bar_i - y_pred_bar_i)
            b_i = n_i * sig2b * (y_bar_i - y_pred_i) / (sig2e + n_i * sig2b)
        else:
            b_i = 0
        b_hat.append(b_i)
    return np.array(b_hat)

def custom_train_generator_lmmnn(train_generator, epochs):
    count = 0
    while True:
        if count == train_generator.n * epochs:
            train_generator.reset()
            break
        count += train_generator.batch_size
        data = train_generator.next()
        imgs = data[0]
        y_true = data[1][:, 0]
        Z = data[1][:, 1]
        yield [imgs, y_true, Z], None

def custom_valid_generator_lmmnn(valid_generator, epochs):
    count = 0 
    while True:
        if count == valid_generator.n * epochs:
            valid_generator.reset()
            break
        count += valid_generator.batch_size
        data = valid_generator.next()
        imgs = data[0]
        y_true = data[1][:, 0]
        Z = data[1][:, 1]
        yield [imgs, y_true, Z], None

def custom_test_generator_lmmnn(test_generator, epochs):
    count = 0 
    while True:
        if count == test_generator.n * epochs:
            test_generator.reset()
            break
        count += test_generator.batch_size
        data = test_generator.next()
        imgs = data[0]
        y_true = data[1][:, 0]
        Z = data[1][:, 1]
        yield [imgs, y_true, Z], None

def custom_train_generator_embed(train_generator, epochs):
    count = 0 
    while True:
        if count == train_generator.n * epochs:
            train_generator.reset()
            break
        count += train_generator.batch_size
        data = train_generator.next()
        imgs = data[0]
        y_true = data[1][:, 0]
        Z = data[1][:, 1]
        yield [imgs, Z], y_true

def custom_valid_generator_embed(valid_generator, epochs):
    count = 0 
    while True:
        if count == valid_generator.n * epochs:
            valid_generator.reset()
            break
        count += valid_generator.batch_size
        data = valid_generator.next()
        imgs = data[0]
        y_true = data[1][:, 0]
        Z = data[1][:, 1]
        yield [imgs, Z], y_true

def custom_test_generator_embed(test_generator, epochs):
    count = 0 
    while True:
        if count == test_generator.n * epochs:
            test_generator.reset()
            break
        count += test_generator.batch_size
        data = test_generator.next()
        imgs = data[0]
        y_true = data[1][:, 0]
        Z = data[1][:, 1]
        yield [imgs, Z], y_true

def factors(n):    # (cf. https://stackoverflow.com/a/15703327/849891)
    j = 2
    while n > 1:
        for i in range(j, int(np.sqrt(n+0.05)) + 1):
            if n % i == 0:
                n /= i ; j = i
                yield i
                break
        else:
            if n > 1:
                yield n; break

def get_batchsize_steps(n):
    factors_n = list(factors(n))
    if len(factors_n) > 1:
        batch_size = factors_n[-2]
    else:
        batch_size = 1
    steps = n // batch_size
    return batch_size, steps

def get_generators(images_df, images_dir, train_samp_subj, valid_samp_subj, test_samp_subj, batch_size, reg_type):
    train_datagen = ImageDataGenerator(rescale = 1./255) # preprocessing_function = preprocess_input # for inception
    valid_datagen = ImageDataGenerator(rescale = 1./255) # preprocessing_function = preprocess_input # for inception
    test_datagen = ImageDataGenerator(rescale = 1./255) # preprocessing_function = preprocess_input # for inception
    if reg_type == 'ignore':
        y_cols = ['age']
    else:
        y_cols = ['age', 'subject_id2']
    train_generator = train_datagen.flow_from_dataframe(
        images_df[images_df['subject_id2'].isin(train_samp_subj)],
        directory = images_dir,
        x_col = 'image_id',
        y_col = y_cols,
        target_size = (IMG_WIDTH, IMG_WIDTH),
        class_mode = 'raw',
        batch_size = batch_size,
        shuffle = True,
        validate_filenames = False
    )
    valid_generator = valid_datagen.flow_from_dataframe(
        images_df[images_df['subject_id2'].isin(valid_samp_subj)],
        directory = images_dir,
        x_col = 'image_id',
        y_col = y_cols,
        target_size = (IMG_WIDTH, IMG_WIDTH),
        class_mode = 'raw',
        batch_size = batch_size,
        shuffle = False,
        validate_filenames = False
    )
    test_generator = test_datagen.flow_from_dataframe(
        images_df[images_df['subject_id2'].isin(test_samp_subj)],
        directory = images_dir,
        x_col = 'image_id',
        y_col = y_cols,
        target_size = (IMG_WIDTH, IMG_WIDTH),
        class_mode = 'raw',
        batch_size = batch_size,
        shuffle = False,
        validate_filenames = False
    )
    return train_generator, valid_generator, test_generator

def reg_nn_ignore(train_generator, valid_generator, test_generator, n_cats, epochs, patience):
    model = cnn_ignore()
    model.compile(loss='mse', optimizer='adam')
    callbacks = [EarlyStopping(monitor='val_loss', patience=epochs if patience is None else patience),
        CSVLogger(logger.handlers[1].baseFilename, append=True)]
    model.fit(train_generator, validation_data = valid_generator, epochs=epochs, callbacks=callbacks, verbose=1)
    y_pred = model.predict(test_generator, verbose=1).reshape(test_generator.n)
    return y_pred, (None, None)

def reg_nn_lmm(train_generator, valid_generator, test_generator, n_cats, epochs, patience):
    model = cnn_lmmnn()
    model.compile(optimizer= 'adam')
    
    patience = epochs if patience is None else patience
    callbacks = [EarlyStoppingWithSigmasConvergence(patience=patience), PrintSigmas(),
        CSVLogger(logger.handlers[1].baseFilename, append=True)]
    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = valid_generator.n // valid_generator.batch_size
    model.fit(custom_train_generator_lmmnn(train_generator, epochs), steps_per_epoch = step_size_train,
        validation_data = custom_valid_generator_lmmnn(valid_generator, epochs), validation_steps = step_size_valid,
        epochs=epochs, callbacks=callbacks, verbose=1)
    
    sig2e_est, sig2b_est = model.layers[-1].get_vars()
    
    batch_size_train, steps_train = get_batchsize_steps(train_generator.n)
    train_generator.reset()
    train_generator.batch_size = batch_size_train
    y_pred_tr = model.predict(custom_train_generator_lmmnn(train_generator, 1),
                              steps = steps_train,
                              verbose=1).reshape(train_generator.n)
    y_train = train_generator.labels[:, 0]
    Z_train = train_generator.labels[:, 1].astype(np.int)
    Z_test = test_generator.labels[:, 1].astype(np.int)
    b_hat = calc_b_hat(Z_train, y_train, y_pred_tr, n_cats, sig2e_est, sig2b_est)
    batch_size_test, steps_test = get_batchsize_steps(test_generator.n)
    test_generator.batch_size = batch_size_test
    y_pred = model.predict(custom_test_generator_lmmnn(test_generator, 1),
                           steps = steps_test, verbose=0).reshape(test_generator.n) + b_hat[Z_test]
    return y_pred, (sig2e_est, sig2b_est)

def reg_nn_embed(train_generator, valid_generator, test_generator, n_cats, epochs, patience):
    embed_dim = 10
    model = cnn_embedding(n_cats, embed_dim)
    model.compile(loss='mse', optimizer='adam')
    callbacks = [EarlyStopping(monitor='val_loss', patience=epochs if patience is None else patience),
        CSVLogger(logger.handlers[1].baseFilename, append=True)]
    
    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = valid_generator.n // valid_generator.batch_size
    
    model.fit(custom_train_generator_embed(train_generator, epochs), steps_per_epoch = step_size_train,
        validation_data = custom_valid_generator_embed(valid_generator, epochs),
        validation_steps = step_size_valid,
        epochs=epochs, callbacks=callbacks, verbose=1)
    batch_size_test, steps_test = get_batchsize_steps(test_generator.n)
    test_generator.batch_size = batch_size_test
    y_pred = model.predict(custom_test_generator_embed(test_generator, 1),
                           steps = steps_test, verbose=0).reshape(test_generator.n)
    return y_pred, (None, None)
        
def reg_nn(images_df, images_dir, train_samp_subj, valid_samp_subj, test_samp_subj,
    n_cats, batch_size=20, epochs=100, patience=10, reg_type='ignore'):
    train_generator, valid_generator, test_generator = get_generators(
        images_df, images_dir, train_samp_subj, valid_samp_subj, test_samp_subj, batch_size, reg_type)
    
    if reg_type == 'ignore':
        y_pred, sigmas = reg_nn_ignore(train_generator, valid_generator, test_generator, n_cats, epochs, patience)
    elif reg_type == 'lmm':
        y_pred, sigmas = reg_nn_lmm(train_generator, valid_generator, test_generator, n_cats, epochs, patience)
    else:
        y_pred, sigmas = reg_nn_embed(train_generator, valid_generator, test_generator, n_cats, epochs, patience)
    y_test = test_generator.labels[:, 0]
    mse = np.mean((y_pred - y_test)**2)
    mae = np.mean(np.abs(y_pred - y_test))
    return mse, mae, sigmas

def iterate_reg_types(images_df, images_dir, res_df, counter, n_cats, train_samp_subj, valid_samp_subj, test_samp_subj,
    out_file, batch_size, epochs, patience):
        mse_lmm, mae_lmm, sigmas = reg_nn(images_df, images_dir, train_samp_subj, valid_samp_subj, test_samp_subj, n_cats,
            batch_size=batch_size, epochs=epochs, patience=patience, reg_type='lmm')
        logger.info(' finished lmm, mse: %.2f, mae: %.2f' % (mse_lmm, mae_lmm))
        mse_ig, mae_ig, _ = reg_nn(images_df, images_dir, train_samp_subj, valid_samp_subj, test_samp_subj, n_cats,
            batch_size=batch_size, epochs=epochs, patience=patience, reg_type='ignore')
        logger.info(' finished ignore, mse: %.2f, mae: %.2f' % (mse_ig, mae_ig))
        mse_em, mae_em, _ = reg_nn(images_df, images_dir, train_samp_subj, valid_samp_subj, test_samp_subj, n_cats,
            batch_size=batch_size, epochs=epochs, patience=patience, reg_type='embed')
        logger.info(' finished embed, mse: %.2f, mae: %.2f' % (mse_em, mae_em))
        res_df.loc[next(counter)] = ['ignore', mse_ig, mae_ig, np.nan, np.nan]
        res_df.loc[next(counter)] = ['lmm', mse_lmm, mae_lmm, sigmas[0], sigmas[1]]
        res_df.loc[next(counter)] = ['embed', mse_em, mae_em, np.nan, np.nan]
        mse_dec = 100 * (mse_lmm - mse_ig) / mse_ig
        logger.info('mse change from mse_ig: %.2f%%' % mse_dec)
        res_df.to_csv(out_file)

class Count:
    curr = 0

    def __init__(self, startWith=None):
        if startWith is not None:
            Count.curr = startWith - 1

    def gen(self):
        while True:
            Count.curr += 1
            yield Count.curr
    
    def __call__(self):
        return Count.curr

def ukb_simulation(out_file, params):
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # for turning GPU off
    images_df = pd.read_csv(params['images_df_path'])
    images_dir = params['images_dir']
    images_df['age'] = images_df['age'].astype(np.float64)
    res_df = pd.DataFrame(columns=['exp_type', 'mse', 'mae', 'sigma_e_est', 'sigma_b_est'])
    counter = Count().gen()
    n_cats = images_df['subject_id2'].max() + 1
    for i in range(params['n_iter']):
        logger.info('iteration: %d' % i)
        train_samp_subj, valid_samp_subj, test_samp_subj = sample_split(i, n_cats)
        iterate_reg_types(images_df, images_dir, res_df, counter, n_cats, train_samp_subj, valid_samp_subj, test_samp_subj,
            out_file, batch_size=params['batch'], epochs=params['epochs'], patience=params['patience'])
