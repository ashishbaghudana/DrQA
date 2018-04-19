#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""

import logging
import numpy as np
import scipy.sparse as sp

from multiprocessing.pool import ThreadPool
from functools import partial

from . import utils
from . import DEFAULTS
from .. import tokenizers

logger = logging.getLogger(__name__)


class Doc2VecCDocRanker(object):
    """Loads a pre-calculated matrix of document vectors and word vectors
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, doc_vectors=None, word_vectors=None, document_ids=None, strict=True):
        """
        Args:
            doc_vectors: path to document vectors file
            word_vectors: path to word vectors file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        doc_vectors = doc_vectors or DEFAULTS['doc_vectors']
        word_vectors = word_vectors or DEFAULTS['word_vectors']
        document_ids = document_ids or DEFAULTS['document_ids']
        logger.info('Loading %s' % doc_vectors)
        logger.info('Loading %s' % word_vectors)
        self.documents = utils.load_document_vectors(doc_vectors)
        self.word_vectors = utils.load_word_vectors(word_vectors)
        self.id2index, self.index2id = utils.load_document_ids(document_ids)
        self.tokenizer = tokenizers.get_class('spacy')()
        self.num_docs = len(self.documents)
        self.strict = strict

    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        return self.id2index[doc_id]

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        return self.index2id[doc_index]

    def closest_docs(self, query, k=1):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        spvec = self.text2vec(query)
        dist_2 = np.sum((self.documents - spvec)**2, axis=1)
        doc_ids = [np.argmin(dist_2) + 1]
        doc_scores = [dist_2[doc_ids[0]]]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=utils.filter_ngram)

    def text2vec(self, query):
        """Create a vector representation of the query using word vectors"""
        words = utils.normalize(query)
        tokens = self.tokenizer.tokenize(words).words(uncased=True)
        vector = np.zeros((300,))
        count = 0
        for token in tokens:
            if token in self.word_vectors:
                vector += self.word_vectors[token]
                count += 1
        return vector / count
