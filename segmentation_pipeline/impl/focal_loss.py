# MIT License
#
# Copyright (c) 2017 Muhammed Kocabas
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE
import tensorflow as tf
from keras import backend as K
'''
Compatible with tensorflow backend
'''

# def focal_loss(gamma=2, alpha=.25):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return K.clip(-K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0)),-100,500)/5
#     return focal_loss_fixed
# def focusloss(ytrue,ypred):
#     gamma = 2.
#     ytrue = K.flatten(ytrue)
#     ypred = K.flatten(ypred)
#     loss = ytrue*K.log(ypred+K.epsilon())*(1-ypred+K.epsilon())*gamma +  (1-ytrue)*K.log(1-ypred+K.epsilon())*(ypred+K.epsilon())*gamma
#     return -K.mean(loss)
def focal_loss(y_true, y_pred):
    gamma=0.75
    alpha=0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
# def focal_loss2(gamma=1.5):
#     def focal_loss(input, target):
#         max_val =K.clip (-input)
#         loss = input - input * target + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))
#         invprobs = K.log(K.sigmoid(-input * (target * 2 - 1)))
#         loss = K.exp(invprobs * gamma) * loss
#         return K.mean(loss)
#     return focal_loss