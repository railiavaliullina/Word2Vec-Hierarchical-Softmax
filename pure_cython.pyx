#!/usr/bin/env python
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

import cython
import numpy as np
cimport numpy as np
import time

from libc.math cimport exp
from libc.string cimport memset

from cpython cimport PyCapsule_GetPointer
import scipy.linalg.blas as cblas


# y += alpha * x
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

# dot(x, y); return value should be `float`, but it only works with `double` (?!)
ctypedef double (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil

cdef saxpy_ptr saxpy=<saxpy_ptr>PyCapsule_GetPointer(cblas.saxpy._cpointer, NULL)
cdef sdot_ptr sdot=<sdot_ptr>PyCapsule_GetPointer(cblas.sdot._cpointer, NULL)

REAL = np.float32
ctypedef np.float32_t REAL_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void fast_sentence(
    np.uint32_t[::1] word_nodes, np.uint8_t[::1] word_code, unsigned long int codelen,
    REAL_t[:, ::1] _w0, REAL_t[:, ::1] _w1, int size,
    np.uint32_t context_word_index, REAL_t alpha, REAL_t[::1] _neu1e) nogil:
    cdef long long i, j
    cdef long long row1 = context_word_index * size, row2
    cdef REAL_t f, g
    cdef REAL_t *w0 = &_w0[0, 0]
    cdef REAL_t *w1 = &_w1[0, 0]
    cdef REAL_t *neu1e = &_neu1e[0]

    for i in range(size):
        neu1e[i] = 0

    for j in range(codelen):
        f = 0
        row2 = word_nodes[j] * size
        for i in range(size):
            f += w1[row1 + i] * w0[row2 + i]
        f = 1 / (1 + exp(-f))
        g = (1 - word_code[j] - f) * alpha

        for i in range(size):
            neu1e[i] += g * w0[row2 + i]

        for i in range(size):
            w0[row2 + i] += g * w1[row1 + i]

    for i in range(size):
        w1[row1 + i] += neu1e[i]


def train_sentence(model, sentence, alpha):
    """
    sentence is a list of objects
    word.nodes - np.array of indexes of nodes in path of huffman tree
    word.code - np.array of 0 and 1 (left, right directions)
    """
    neu1e = np.empty(model.hidden_size, dtype=REAL)


    len_sentence = len(sentence)
    lr_step = (alpha - 0.0001) / len_sentence

    start_time_total = time.time()
    start_time = time.time()
    for word_pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip

        if word_pos % 20000 == 0:
            print(f'word_pos: {word_pos}/{len_sentence}, time: {time.time() - start_time} s, alpha: {alpha}')
            start_time = time.time()

        # now go over all words from the (reduced) window, predicting each one in turn
        start = max(0, word_pos - model.window)  #  + reduced_window
        end = min(len_sentence, word_pos + model.window + 1)
        for context_pos, context_word in enumerate(sentence[start : end], start):

            if context_pos == word_pos or context_word is None:
                # don't train on OOV words and on the `word` itself
                continue

            fast_sentence(word.nodes, word.code, len(word.nodes), model.w0, model.w1, model.hidden_size,
                          context_word.index, alpha, neu1e)

        alpha -= lr_step
    print(f'Total time: {time.time() - start_time_total} s')
    return model
