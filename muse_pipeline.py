# first train a muse model
import io
import csv
import pickle
import codecs
import numpy as np
import pandas as pd
from scipy import spatial
from gensim.models import KeyedVectors
from pca_scatter_plot import pca_scatter_plot

to_write = True
# different processing scenarios correspond to folders
folders = [
    "./",
    "no_refinement/",
    "refinement_1000/",
    "translated/",
    "unsupervised_translation/",
    "unsupervised_translation_2/",
    "supervised/",
    "vecmap_unsupervised_default/",
    "combined_cosine/",
    "vecmap_semi_default/",
    "vecmap_supervised/",
    "averaged_word2vec/",
    "averaged_word2vec_common_space/",
    "averaged_word2vec_vecmap/",
    "averaged_word2vec_uncommon_space/",
    "last_unsupervised_muse_av/",
    "doc2vec_english_vectors/",
    "vecmap-70/",
    "fasttext_vectors/",
    "fasttext-vecmap/",
    "pca-10/",
    "pca-20/",
    "fasttext-vecmap-1/",
]

limits = ["70", "10", "30", "50"]
rank_limit = ""
rank_limit = "-460"
for limit in limits:
    folders.append("vecmap-{}-semi{}/".format(limit, rank_limit))
    folders.append("vecmap-{}-full{}/".format(limit, rank_limit))
    folders.append("muse-{}{}/".format(limit, rank_limit))

pca_scatter_plotpca_scatter_plot(
    (maryland_vectors, okpd_vectors),
    labels=["maryland", "okpd"], tsne_plot=True, my_plot=False)


def match_dicts(conversion_dict, source_names, target_names, filename,
                scores_provided=False, to_write=False):
    # maryland2okpd, marpca_scatter_plotyland_dict,
    # okpd_dict, maryland2okpd_dict_file
    matched_pairs = []
    scores = []
    if to_write:
        f = open(filename, "w")
        f.write("")
        f.flush()
        f.close()
        f = open(filename, "a")
        f.write(
            "source_code\ttarget_code\tsource_name\ttarget_name\t"
            "target_score\n")
    for source_code in conversion_dict:
        target_score = ""
        target_code = ""
        if scores_provided:
            target_code = conversion_dict[source_code][1]
            try:
                target_score = round(conversion_dict[source_code][0], 3)
            except Exception as ex:
                print(ex)
        else:
            target_code = conversion_dict[source_code]
        source_name = ""
        target_name = ""
        if source_code in source_names:
            source_name = source_names[source_code]
            source_name = " ".join(source_name.split())
        else:
            print(source_code)
        if target_code in target_names:
            target_name = target_names[target_code]
            target_name = " ".join(target_name.split())
        else:
            print(target_code)
        source_code = str(source_code)
        target_code = str(target_code)
        matched_pairs.append([source_code, target_code])
        scores.append(target_score)
        if to_write:
            f.write(
                "{}\t{}\t{}\t{}\t{}\n".format(
                    source_code, target_code, source_name, target_name,
                    target_score))
    return matched_pairs, scores


def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(
            emb_path, 'r', encoding='utf-8',
            newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


def write_top_indices(write_pairs, scores, filename, top_percent=10):
    f = open(filename, "w")
    f.write("")
    f.flush()
    f.close()
    f = open(filename, "a")
    sorted_indices = np.argsort(scores)[::-1]
    top_limit = int(top_percent / 100 * len(sorted_indices))
    for i, sorted_i in enumerate(sorted_indices):
        if i > top_limit:
            break
        source_code, target_code = write_pairs[sorted_i]
        f.write("{} {}\n".format(source_code, target_code))


def get_nn(
        word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5, verbose=True):
    if verbose:
        print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (
        tgt_emb / np.linalg.norm(
            tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    output = []
    for i, idx in enumerate(k_best):
        if verbose:
            print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
        output.append([scores[idx], tgt_id2word[idx]])
    return output


def create_dict(
        name, src_embeddings, tgt_embeddings, src_id2word, tgt_id2word):
    # For each word there is a list
    # where list[0] is the score
    # and list[1] is the name
    source_2_target = dict()
    for word_id in src_id2word:
        print(
            "{0:05d}".format(len(src_id2word) - word_id), end="\r")
        src_word = src_id2word[word_id]
        closest_word = get_nn(
            src_word, src_embeddings, src_id2word, tgt_embeddings,
            tgt_id2word, K=5, verbose=False)
        # source_2_target[src_word] = closest_word[0][1]
        source_2_target[src_word] = closest_word[0]
    print("Dict len is", len(source_2_target))
    pickle.dump(source_2_target, open(name, "wb"))


def create_dict_cosine(name, src_embeddings, tgt_embeddings):
    categories_dict = dict()
    tgt_categories = list(tgt_embeddings.vocab)
    for src_category_i, src_category in src_embeddings.vocab:
        print(src_category_i, end="\r")
        similarities = np.zeros(len(tgt_categories))
        for tgt_category_i, tgt_category in enumerate(tgt_categories):
            similarities[tgt_category_i] = 1 - spatial.distance.cosine(
                src_embeddings[src_category], tgt_embeddings[tgt_category])
        most_similar = np.argmax(similarities)
        categories_dict[src_category] = tgt_categories[most_similar]


def main(folder, to_write):
    print(folder)
    maryland_vectors_file = folder + "vectors-maryland.txt"
    okpd_vectors_file = folder + "vectors-okpd.txt"
    maryland2okpd_file = folder + "maryland2okpd.pcl"
    okpd2maryland_file = folder + "okpd2maryland.pcl"
    maryland2okpd_dict_file = folder + "maryland2okpd_dict.csv"
    okpd2maryland_dict_file = folder + "okpd2maryland_dict.csv"
    if "muse" in folder:
        muse_dico_file = folder + "muse_dico.pcl"
        src_dico_i2word_file = folder + "src_dico_i2word.pcl"
        tgt_dico_i2word_file = folder + "tgt_dico_i2word.pcl"
        maryland2okpd_muse_dico_file = folder + "maryland2okpd_muse_dico.csv"

    maryland = KeyedVectors.\
        load_word2vec_format(maryland_vectors_file)
    okpd = KeyedVectors.\
        load_word2vec_format(okpd_vectors_file)
    try:
        maryland_vectors = maryland.vectors
    except Exception as ex:
        maryland_vectors = maryland.syn0
    try:
        okpd_vectors = okpd.vectors
    except Exception as ex:
        okpd_vectors = okpd.syn0
    pca_scatter_plot(
        (maryland_vectors, okpd_vectors),
        labels=["maryland", "okpd"], tsne_plot=False, my_plot=False,
        show_figures=False, path=folder)
    src_path = maryland_vectors_file
    tgt_path = okpd_vectors_file
    nmax = 50000  # maximum number of word embeddings to load
    src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax)
    tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path, nmax)
    create_dict(
        maryland2okpd_file, src_embeddings,
        tgt_embeddings, src_id2word, tgt_id2word)
    create_dict(
        okpd2maryland_file, tgt_embeddings,
        src_embeddings, tgt_id2word, src_id2word)

    maryland2okpd = pickle.load(open(maryland2okpd_file, "rb"))
    okpd2maryland = pickle.load(open(okpd2maryland_file, "rb"))
    maryland_names = dict()
    okpd_names = dict()
    f = codecs.open("okpd2.csv", "r", "cp1251")
    reader = csv.reader(f, delimiter=";")
    okpd_dict = {l[3]: l[1] for l in reader}
    years = list(range(2012, 2018))
    maryland_dict = dict()
    for year in years:
        df = pd.read_csv(
            "eMaryland_Marketplace_Bids_-_Fiscal_Year_{}.csv".format(year))
        class_descriptions = df["NIGP Class with Description"].\
            str.replace("^[0-9]+ - ", "").str.strip() + " "
        item_descriptions = df["NIGP Class Item with Description"].\
            str.replace("^[0-9]+ - \d+ :", "").str.strip()
        class_descriptions = class_descriptions + item_descriptions
        codes = df["NIGP 5 digit code"]
        codes_dict = dict(zip(codes, class_descriptions))
        if year == 2012:
            maryland_dict = codes_dict
        else:
            maryland_dict.update(codes_dict)

    matched_pairs, scores = match_dicts(
        maryland2okpd, maryland_dict, okpd_dict, maryland2okpd_dict_file,
        scores_provided=True, to_write=to_write)
    matched_pairs_okpd, scores_okpd = match_dicts(
        okpd2maryland, okpd_dict, maryland_dict, okpd2maryland_dict_file,
        scores_provided=True, to_write=to_write)
    if not to_write:
        top_n = [10, 30, 50, 70]
        for n in top_n:
            write_top_indices(
                matched_pairs, scores, "maryland_dict_{}".format(n),
                top_percent=n)
    if "muse" in folder:
        muse_dico = pickle.load(open(muse_dico_file, "rb"))
        src_dico = pickle.load(open(src_dico_i2word_file, "rb"))
        tgt_dico = pickle.load(open(tgt_dico_i2word_file, "rb"))

        muse_dico_names = {
            src_dico[k]: tgt_dico[muse_dico[k]] for k in muse_dico}

        match_dicts(
            muse_dico_names, maryland_dict, okpd_dict,
            maryland2okpd_muse_dico_file, scores_provided=False, to_write=True
        )


if not folders:
    main(folder, to_write)
else:
    for folder in folders:
        main(folder, to_write)
