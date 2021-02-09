#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from multiprocessing import cpu_count
from six import string_types

from gensim.models import Word2Vec
from gensim.models.word2vec import Vocab

import logging


# In[ ]:


class Skipgram(Word2Vec):#Skipgram(used Skipgram code from original implementation)

    def __init__(self, vocabulary_counts=None, **kwargs):

        self.vocabulary_counts = None

        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["workers"] = kwargs.get("workers", cpu_count())
        kwargs["size"] = kwargs.get("size", 128)#size of embedding vector
        kwargs["sentences"] = kwargs.get("sentences", None)
        kwargs["window"] = kwargs.get("window", 10)
        kwargs["sg"] = 1#skip_gram
        kwargs["hs"] = 1#hierachical softmax

        if vocabulary_counts != None:
          self.vocabulary_counts = vocabulary_counts

        super(Skipgram, self).__init__(**kwargs)

