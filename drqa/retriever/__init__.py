#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from .. import DATA_DIR

DEFAULTS = {
    'db_path': os.path.join(DATA_DIR, 'medline/docs.db'),
    'tfidf_path': os.path.join(
        DATA_DIR,
        'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
    ),
    'doc_vectors': os.path.join(DATA_DIR, 'medline/docvectors.txt'),
    'word_vectors': os.path.join(DATA_DIR, 'medline/wordvectors.txt'),
    'document_ids': os.path.join(DATA_DIR, 'medline/document_ids.txt'),
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'doc2vecc':
        return Doc2VecCDocRanker
    if name == 'sqlite':
        return DocDB
    raise RuntimeError('Invalid retriever class: %s' % name)


from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker
from .doc2vecc_doc_ranker import Doc2VecCDocRanker

