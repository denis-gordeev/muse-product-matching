import gc
# import csv
import pandas as pd
# import codecs
import logging
# import numpy as np
# from collections import Counter
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from utils_d.utils import text_pipeline, texts_pipeline_multiprocessing
from pca_scatter_plot import pca_scatter_plot

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class LinesIterator(object):
    def __init__(self, strings_container, tags_container):
        self.strings_container = strings_container
        self.tags_container = tags_container

    def __iter__(self):
        # for l in open("okpds_cbuster.csv"):
        for l_i, l in enumerate(self.strings_container):
            tags = self.tags_container[l_i]
            output = TaggedDocument(words=l, tags=tags)
            # l = nltk.word_tokenize(l)
            # l = [w.lower() for w in l]
            yield output


f = open("okpds3_unique")
r = f.readlines()
r = [l.strip().split("\t") for l in r]
x = [l[0] for l in r]
y = [l[1] for l in r]

del r
gc.collect()


def filter_y_by_x(x, y):
    for l_i, l in enumerate(x):
        if not l:
            y[l_i] = None
    x = [l for l in x if l]
    y = [l for l in y if l]
    return x, y


def save_category_vectors(
        model, filename, y=None, y_tags_from_model=False, maryland=False):
    if not y_tags_from_model and y:
        y_names = set([l[-1] for l in y])
    else:
        y_names = model.docvecs.doctags.keys()
    if maryland:
        y_names = [t for t in y_names if "-" in t]
    else:
        y_names = [t for t in y_names if len(t.split(".")) == 4]

    vectors = {}

    for name in y_names:
        vectors[name] = model.docvecs[name]
    f = open("doc2vec/doc2vec_category_vectors_{}.txt".format(filename), "w")
    f.write("")
    f.flush()
    f.close()
    f = open("doc2vec/doc2vec_category_vectors_{}.txt".format(filename), "a")
    f.write("{} {}\n".format(len(vectors), model.vector_size))
    for v in vectors:
        line = "{} {}\n".format(v, " ".join(vectors[v].astype(str)))
        f.write(line)
    f.close()

x = texts_pipeline_multiprocessing(x)
x, y = filter_y_by_x(x, y)

for l_i, l in enumerate(y):
    l = l.split(".")
    tags = []
    tag = []
    for el in l:
        tag.append(el)
        tags.append(".".join(tag))
    y[l_i] = tags

model = Doc2Vec(
    LinesIterator(x, y), size=300, window=10, min_count=2, workers=4)

model.save("doc2vec/doc2vec_okpd")
save_category_vectors(model, "okpd")

x = []
y = []

gc.collect()

years = list(range(2012, 2018))

for year in years:
    df = pd.read_csv(
        "eMaryland_Marketplace_Bids_-_Fiscal_Year_{}.csv".format(year))
    df = df[pd.notna(df['Description'])]
    df = df[pd.notna(df['NIGP Class'])]
    df = df[pd.notna(df['NIGP Class Item'])]
    x += list(df["Description"].values)
    nigp_class = list(df['NIGP Class'].astype(int).astype(str).values)
    nigp_class_item = df['NIGP Class'].astype(int).astype(str) + "-" +\
        df['NIGP Class Item'].astype(int).astype(str)
    nigp_class_item = list(nigp_class_item.values)
    y += list(zip(nigp_class, nigp_class_item))
    x += list(df["Bid Attachment Description"].values)
    y += list(zip(nigp_class, nigp_class_item))

x = texts_pipeline_multiprocessing(x)
x, y = filter_y_by_x(x, y)


model = Doc2Vec(
    LinesIterator(x, y), size=300, window=10, min_count=2, workers=4)

model.save("doc2vec/doc2vec_maryland")

save_category_vectors(model, "maryland", maryland=True)
maryland = KeyedVectors.\
    load_word2vec_format("doc2vec/doc2vec_category_vectors_maryland.txt")
okpd = KeyedVectors.\
    load_word2vec_format("doc2vec/doc2vec_category_vectors_okpd.txt")
maryland_vectors = maryland.vectors
okpd_vectors = okpd.vectors

pca_scatter_plot(
    (maryland_vectors, okpd_vectors),
    labels=["maryland", "okpd"], tsne_plot=True, my_plot=False)
