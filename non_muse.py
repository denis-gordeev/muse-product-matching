import re
import pandas as pd
import numpy as np
import pickle

from scipy import spatial
from gensim.models import KeyedVectors, Doc2Vec
from nltk import word_tokenize

from utils import read_nigp_csv, get_averaged_vectors, load_okpd_dict

remove_strings = [
    "не включенные в другие группировки",
    "\(Not Otherwise Classified\)",
    "кроме .*",
    "\(See .*?\)",
    "Also See .*",
    "\(not .*\)",
    "\(for .* see\)",
    "\(except .*\)",
    "кроме .*"

]

remove_strings = [l.lower() for l in remove_strings]
remove_strings = "|".join(remove_strings)
HIERARCHICAL = True
LIMIT = False
WORD2VEC_MATCHING = True
TRANSLATE = False
if HIERARCHICAL:
    LIMIT = False
create_averaged = False
create_averaged_english_only = False
write_dict = False
create_doc2vec = True

okpd_dict = load_okpd_dict()


try:
    names_eng = pickle.load(open("okpd_eng.pcl", 'rb'))
except Exception as ex:
    print(ex)
try:
    names_eng_yandex = pickle.load(open("okpd_eng_yandex.pcl", 'rb'))
except Exception as ex:
    print(ex)


okpd_eng = dict()
for okpd in okpd_dict:
    if LIMIT:
        if len(okpd.split(".")) < 4:
            continue
    ru_name = okpd_dict[okpd]
    if not ru_name:
        okpd_eng[okpd] = ""
    elif ru_name in names_eng:
        okpd_eng[okpd] = names_eng[ru_name].text
    elif ru_name in names_eng_yandex:
        okpd_eng[okpd] = names_eng_yandex[ru_name]
    else:
        print(ru_name)
if not TRANSLATE:
    okpd_eng = okpd_dict


"""
nigp_0 =
    {str(int(m)):
     nigp[m] for m in nigp if str(int(m)).isdigit()}
"""
nigp = read_nigp_csv()


def clean_file(filename):
    with open(filename, "w") as f:
        f.write("")
        f.flush()
    f = open(filename, "a")
    return f


def filter_dict(source_dict):
    source_dict = {k: source_dict[k] for k in source_dict
                   if not pd.isnull(k) and not pd.isnull(source_dict[k]) and
                   re.findall("[0-9]|-", k)}
    return source_dict


def clean_dict_strings(source_dict):
    for k in source_dict:
        value = source_dict[k].lower()
        value = re.sub(remove_strings, "", value)
        source_dict[k] = value
    return source_dict


def create_limit_dicts(source_dict, delimiter=".", limit_range=range(1, 2)):
    limit_dicts = []
    limit_dicts_inverse = []
    for i in limit_range:
        # Maryland
        if delimiter == "-":
            if i == 1:
                limit_dict = {
                    k.split("-")[0]: source_dict[k].split("::")[0].strip()
                    for k in source_dict}
            else:
                limit_dict = {
                    k: source_dict[k].split("::")[1].strip()
                    for k in source_dict}
        # OKPD
        else:
            limit_dict = {
                o: source_dict[o] for o in source_dict
                if len(o.split(delimiter)) == i}
        limit_dicts.append(limit_dict)
    return limit_dicts


def process_dict(source_dict, delimiter=".", limit_range=range(1, 4)):
    source_dict = filter_dict(source_dict)
    if HIERARCHICAL:
        limit_dicts = create_limit_dicts(source_dict, delimiter, limit_range)
    else:
        limit_dicts = []
    source_dict = clean_dict_strings(source_dict)
    source_dict_inverse = dict()
    for k in source_dict:
        description = source_dict[k]
        if description in source_dict_inverse:
            source_dict_inverse[description].append(k)
        else:
            source_dict_inverse[description] = [k]
    names = list({source_dict[k] for k in source_dict})
    names_split = [set(word_tokenize(name)) for name in names]
    return source_dict, source_dict_inverse, names, names_split, limit_dicts


nigp, nigp_inverse, nigp_names,\
    nigp_split, limit_dicts = process_dict(nigp, delimiter="-",
                                           limit_range=range(1, 3))
if HIERARCHICAL:
    nigp_limits = limit_dicts
    nigp_limits = [clean_dict_strings(d) for d in nigp_limits]

okpd_eng, okpd_inverse, okpd_names, okpd_split, limit_dicts = process_dict(
    okpd_eng)
if HIERARCHICAL:
    okpd_limits = limit_dicts
    okpd_limits = [clean_dict_strings(d) for d in okpd_limits]

nigp_closest = []


def compare_lists_of_sets(first_list, second_list, return_all=False):
    closest_sets = []
    for m_i, m in enumerate(first_list):
        print("\t", m_i, end='\r')
        if WORD2VEC_MATCHING:
            intersections = []
            for o in second_list:
                try:
                    distance = 1 - spatial.distance.cosine(m, o)
                except Exception as ex:
                    distance = 0
                intersections.append(distance)

        else:
            intersections = [len(m.intersection(o)) for o in second_list]
        max_intersection = np.argmax(intersections)
        max_value = intersections[max_intersection]
        if not return_all:
            closest_sets.append([max_intersection, max_value])
        else:
            max_intersections = np.argwhere(np.array(intersections) ==
                                            max_value)
            if max_intersections.size > 0:
                max_intersections = [l[0] for l in max_intersections]
            closest_sets.append([max_intersections, max_value])
    return closest_sets


# Sum of intersection length of common words in okpd and nigp
"""
Leave only top level if no intersection
nigp -> okpd
0 -> 1
0 -> 2
0 -> 3
1 -> 3
"""
if WORD2VEC_MATCHING:
    if TRANSLATE:
        path = '/home/denis/ranepa/embeddings/'\
            'GoogleNews-vectors-negative300-SLIM.bin'
        model = KeyedVectors.load_word2vec_format(path, binary=True)
        model_ru = model
        model_en = model
    else:
        russian_fasttext = "/home/denis/ranepa/articles/muse-classifier/"\
            "muse_embeddings/wiki.multi.ru.vec"
        model_ru = KeyedVectors.load_word2vec_format(russian_fasttext)
        english_fasttext = "/home/denis/ranepa/articles/muse-classifier/"\
            "muse_embeddings/wiki.multi.en.vec"
        model_en = KeyedVectors.load_word2vec_format(english_fasttext)

if HIERARCHICAL:
    last_okpd = 0
    best_classes = []
    old_codes = []
    old_names = []
    for m_i, m_limit in enumerate(nigp_limits):
        m_inverse = {m_limit[k]: k for k in m_limit}
        m_codes = [k for k in m_limit]
        print("\t", m_i, m_codes[0])
        m_names = [m_limit[k] for k in m_codes]
        if m_i == 0:
            best_classes = [None] * len(m_names)
        else:
            best_classes = [[i for i in range(len(old_codes))
                             if m.startswith(old_codes[i])][0]
                            for m in m_codes]
        m_split = [set(word_tokenize(k)) for k in m_names]
        if WORD2VEC_MATCHING:
            m_split = get_averaged_vectors(model_en, m_split)
        old_names = m_names
        old_codes = m_codes
        for o_i, o_limit in enumerate(okpd_limits):
            print("\t", o_i)
            o_inverse = {o_limit[k]: k for k in o_limit}
            o_names = [k for k in o_inverse]
            o_split = [set(word_tokenize(k)) for k in o_names]
            if WORD2VEC_MATCHING:
                o_split = get_averaged_vectors(model_ru, o_split)
                if o_split[0].shape[0] != 300:
                    raise Exception
            closest_sets = compare_lists_of_sets(m_split, o_split,
                                                 return_all=True)
            for c_i, c in enumerate(closest_sets):
                if c[1] == 0:
                    print(c_i, len([l for l in closest_sets if l[1] == 0]))
            for c_i, c in enumerate(closest_sets):
                new_best_classes = [[o_inverse[o_names[cl_name]]
                                     for cl_name in c[0]], c[1]]
                if o_i == 0:
                    best_classes[c_i] = new_best_classes
                else:
                    old_classes = best_classes[c_i]
                    if new_best_classes[1] >= old_classes[1]:
                        intersection = [n for n in new_best_classes[0] if
                                        any(n.startswith(old_c) for
                                            old_c in old_classes[0])]
                        if intersection:
                            best_classes[c_i] = [intersection,
                                                 new_best_classes[1]]
                        # else:
                        #     best_classes[c_i][0] += new_best_classes[0]
                        #     best_classes[c_i][1] = new_best_classes[1]
    max_values = [l[1] for l in best_classes]
    best_classes = [b[0][0] for b in best_classes]
    top_indices = np.argsort(max_values)[::-1]
    filename = "matcher_hierarchy2.csv"
    if WORD2VEC_MATCHING:
        filename = "w2v_" + filename
    if not TRANSLATE:
        filename = "aligned_" + filename
    f = clean_file(filename)
    for i in top_indices:
        nigp_code = old_codes[i]
        okpd_code = best_classes[i]
        nigp_name = nigp[nigp_code]
        okpd_name = okpd_dict[okpd_code]
        value = max_values[i]
        f.write(
            "{}\t{}\t{}\t{}\t{}\n".format(
                nigp_code, okpd_code, nigp_name, okpd_name, value))
else:
    nigp_closest = []
    for m_i, m in enumerate(nigp_split):
        print("\t {}".format(m_i), end="\r")
        if WORD2VEC_MATCHING:
            pass
        else:
            pass
        intersections = [
            len(m.intersection(o)) for o in okpd_split]
        max_intersection = np.argmax(intersections)
        max_value = intersections[max_intersection]
        nigp_closest.append([max_intersection, max_value])

    # max_values = [v[1] for v in nigp_closest]

    max_values = [
        np.average([nigp_closest[v_i][1] / len(nigp_split[v_i]),
                   nigp_closest[v_i][1] / len(okpd_split
                                                  [nigp_closest[v_i][0]])])
        for v_i in range(len(nigp_closest))]
    compare_nigp = np.argmax(max_values)
    compare_okpd = nigp_closest[compare_nigp][0]
    print(nigp_names[compare_nigp])
    print(okpd_names[compare_okpd])

top_indices = np.argsort(max_values)[::-1]

# top 10
nigp2okpd_codes = dict()
if write_dict:
    f = clean_file("matcher_limit.csv")
for top_i in top_indices:  # [:int(len(top_indices) / 10)]:
    value = max_values[top_i]
    # if value <= 0.5 and not write_dict:
    #     break
    nigp_name = nigp_names[top_i]
    okpd_name = okpd_names[nigp_closest[top_i][0]]
    nigp_code = nigp_inverse[nigp_name]
    okpd_codes = okpd_inverse[okpd_name]
    if okpd_codes:
        okpd_codes = okpd_codes[:1]
        for okpd_code in okpd_codes:
            ru_name = okpd_dict[okpd_code]
            nigp2okpd_codes[nigp_code] = okpd_code
            if write_dict:
                f.write(
                    "{}\t{}\t{}\t{}\t{}\n".format(
                        nigp_code, okpd_code, nigp_name, ru_name,
                        value))
    if not write_dict:
        print(nigp_name)
        print(okpd_name)
        print(value)
okpd2nigp_codes = {nigp2okpd_codes[k]: k for k in nigp2okpd_codes}


def write_dict_to_txt(filename, src_dict):
    f = open(filename, "w")
    f.write("")
    f.flush()
    f.close()
    with open(filename, 'a') as f:
        for key in src_dict:
            f.write("{} {}\n".format(key, src_dict[key]))


# if write_dict:
#     write_dict_to_txt("nigp2okpd_dict_LIMIT.txt", nigp2okpd_codes)
#     write_dict_to_txt("okpd2nigp_LIMIT.txt", okpd2nigp_codes)


def vectors_to_file(sents_averaged, sents, filename, dict_inverse,
                    nigp=False, vector_size=300):
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
            if nigp:
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

        nigp_averaged = get_averaged_vectors(model, nigp_split)
        okpd_averaged = get_averaged_vectors(model, okpd_split)
    else:
        if create_doc2vec:
            path = "doc2vec/doc2vec_nigp"
            model = Doc2Vec.load(path)
            nigp_docvecs = np.array([model.infer_vector(sent)
                                     for sent in nigp_names])
            okpd_docvecs = np.array([model.infer_vector(sent)
                                     for sent in okpd_names])
            vectors_to_file(
                nigp_docvecs, nigp_names,
                "doc2vec_english_vectors/vectors-nigp.txt",
                nigp_inverse,
                nigp=True,
            )
            vectors_to_file(
                okpd_docvecs, okpd_names,
                "doc2vec_english_vectors/vectors-okpd.txt",
                okpd_inverse,
                nigp=False,
            )
        else:
            path = '/home/denis/ranepa/articles/muse-classifier/'\
                'muse_embeddings/wiki.multi.en.vec'
            model = KeyedVectors.load_word2vec_format(path)
            nigp_averaged = get_averaged_vectors(
                model, nigp_split)
            path = '/home/denis/ranepa/articles/muse-classifier/'\
                'muse_embeddings/wiki.multi.ru.vec'
            model = KeyedVectors.load_word2vec_format(path)
            okpd_averaged = get_averaged_vectors(
                model, okpd_split)

        vectors_to_file(
            nigp_averaged, nigp_names,
            "averaged_word2vec_common_space/vectors-nigp.txt",
            nigp_inverse,
            nigp=True,
        )
        vectors_to_file(
            okpd_averaged, okpd_names,
            "averaged_word2vec_common_space/vectors-okpd.txt",
            okpd_inverse,
            nigp=False,
        )
