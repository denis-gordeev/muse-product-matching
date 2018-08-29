import pandas as pd
import numpy as np
import codecs
import csv
import pickle

from gensim.models import KeyedVectors
from nltk import word_tokenize

LIMIT = True
create_averaged = True
create_averaged_english_only = False
write_dict = False
f = codecs.open("okpd2.csv", "r", "cp1251")
reader = csv.reader(f, delimiter=";")
okpd_dict = {l[3]: l[1] for l in reader}


try:
    names_eng = pickle.load(open("okpd_eng.pcl", 'rb'))
except Exception as ex:
    print(ex)
try:
    names_eng_yandex = pickle.load(open("okpd_eng_yandex.pcl", 'rb'))
except Exception as ex:
    print(ex)


okpd_dict_eng = dict()
for okpd in okpd_dict:
    if LIMIT:
        if len(okpd.split(".")) < 4:
            continue
    ru_name = okpd_dict[okpd]
    if not ru_name:
        okpd_dict_eng[okpd] = ""
    elif ru_name in names_eng:
        okpd_dict_eng[okpd] = names_eng[ru_name].text
    elif ru_name in names_eng_yandex:
        okpd_dict_eng[okpd] = names_eng_yandex[ru_name]
    else:
        print(ru_name)

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

maryland_dict = {k: maryland_dict[k] for k in maryland_dict
                 if not pd.isnull(k)}
maryland_dict = {k: maryland_dict[k] for k in maryland_dict
                 if not pd.isnull(maryland_dict[k])}

maryland_dict = {k: maryland_dict[k].lower() for k in maryland_dict}
okpd_dict_eng = {k: okpd_dict_eng[k].lower() for k in okpd_dict_eng}

# 1 to 1
maryland_dict_inverse = {maryland_dict[k]: k for k in maryland_dict}
maryland_names = [maryland_dict[k] for k in maryland_dict]
okpd_dict_eng_names = list({okpd_dict_eng[k] for k in okpd_dict_eng})
# 1 to many
okpd_dict_eng_inverse = dict()
for k in okpd_dict_eng:
    description = okpd_dict_eng[k]
    if description in okpd_dict_eng_inverse:
        okpd_dict_eng_inverse[description].append(k)
    else:
        okpd_dict_eng_inverse[description] = [k]

maryland_names_split = [set(word_tokenize(name)) for name in maryland_names]
okpd_dict_eng_names_split = [
    set(word_tokenize(name)) for name in okpd_dict_eng_names]

maryland_closest = []

# Sum of intersection length of common words in okpd and maryland
for m_i, m in enumerate(maryland_names_split):
    print("\t {}".format(m_i), end="\r")
    intersections = [len(m.intersection(o)) for o in okpd_dict_eng_names_split]
    max_intersection = np.argmax(intersections)
    max_value = intersections[max_intersection]
    maryland_closest.append([max_intersection, max_value])

# max_values = [v[1] for v in maryland_closest]
max_values = [
    np.average([maryland_closest[v_i][1] / len(maryland_names_split[v_i]),
               maryland_closest[v_i][1] / len(okpd_dict_eng_names_split
                                              [maryland_closest[v_i][0]])])
    for v_i in range(len(maryland_closest))]
compare_maryland = np.argmax(max_values)
compare_okpd = maryland_closest[compare_maryland][0]
print(maryland_names[compare_maryland])
print(okpd_dict_eng_names[compare_okpd])

top_indices = np.argsort(max_values)[::-1]

# top 10
maryland2okpd_codes = dict()
for top_i in top_indices[:int(len(top_indices) / 10)]:
    value = max_values[top_i]
    if value <= 0.5:
        break
    maryland_name = maryland_names[top_i]
    okpd_name = okpd_dict_eng_names[maryland_closest[top_i][0]]
    maryland_code = maryland_dict_inverse[maryland_name]
    okpd_codes = okpd_dict_eng_inverse[okpd_name]
    if okpd_codes:
        for okpd_code in okpd_codes:
            maryland2okpd_codes[maryland_code] = okpd_code
    print(maryland_name)
    print(okpd_name)
    print(value)
okpd2maryland_codes = {maryland2okpd_codes[k]: k for k in maryland2okpd_codes}


def write_dict_to_txt(filename, src_dict):
    f = open(filename, "w")
    f.write("")
    f.flush()
    f.close()
    with open(filename, 'a') as f:
        for key in src_dict:
            f.write("{} {}\n".format(key, src_dict[key]))


if write_dict:
    write_dict_to_txt("maryland2okpd_dict_LIMIT.txt", maryland2okpd_codes)
    write_dict_to_txt("okpd2maryland_dict_LIMIT.txt", okpd2maryland_codes)


def get_averaged_vectors(model, split_sents):
    sents_filtered = [[w for w in sent if w in model.vocab]
                      for sent in split_sents]
    sents_averaged = np.array([np.sum([model[w] for w in sent], axis=0)
                              for sent in sents_filtered])
    sents_lengths = np.array([len(m) for m in sents_filtered])
    sents_averaged = sents_averaged.T / sents_lengths
    sents_averaged = sents_averaged.T
    return sents_averaged


def averaged_vectors_to_file(sents_averaged, sents, filename, dict_inverse,
                             maryland=False, vector_size=300):
    f = open(filename, "w")
    f.write("")
    f.flush()
    f.close()
    f = open(filename, "a")
    elements_number = len([s for s in sents_averaged if s.size == vector_size])
    f.write("{} {}\n".format(elements_number, vector_size))
    done_codes = set()
    for v_i, vector in enumerate(sents_averaged):
        if vector.size == vector_size:
            codes = []
            sent = sents[v_i]
            if maryland:
                codes = [dict_inverse[sent]]
            else:
                codes = dict_inverse[sent]
            for code in codes:
                if code in done_codes:
                    continue
                else:
                    done_codes.add(code)
                line = "{} {}\n".format(code, " ".join(vector.astype(str)))
                f.write(line)
    f.close()


if create_averaged:
    if create_averaged_english_only:
        path = '/home/denis/ranepa/embeddings/'\
            'GoogleNews-vectors-negative300-SLIM.bin'
        model = KeyedVectors.load_word2vec_format(path, binary=True)

        maryland_averaged = get_averaged_vectors(model, maryland_names_split)
        okpd_averaged = get_averaged_vectors(model, okpd_dict_eng_names_split)
    else:

        path = '/home/denis/ranepa/articles/muse-classifier/'\
            'muse_embeddings/wiki.multi.en.vec'
        model = KeyedVectors.load_word2vec_format(path)
        maryland_averaged = get_averaged_vectors(model, maryland_names_split)
        path = '/home/denis/ranepa/articles/muse-classifier/'\
            'muse_embeddings/wiki.multi.ru.vec'
        model = KeyedVectors.load_word2vec_format(path)
        okpd_averaged = get_averaged_vectors(model, okpd_dict_eng_names_split)

    averaged_vectors_to_file(
        maryland_averaged, maryland_names,
        "averaged_word2vec_common_space/vectors-maryland.txt",
        maryland_dict_inverse,
        maryland=True,
    )
    averaged_vectors_to_file(
        okpd_averaged, okpd_dict_eng_names,
        "averaged_word2vec_common_space/vectors-okpd.txt",
        okpd_dict_eng_inverse,
        maryland=False,
    )
