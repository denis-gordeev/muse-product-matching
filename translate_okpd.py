import codecs
import csv
import pickle
import time

from googletrans import Translator
from yandex import Translater


def translate():
    """
    translate Russian okpd category descriptions to English
    """
    f = codecs.open("okpd2.csv", "r", "cp1251")
    reader = csv.reader(f, delimiter=";")
    okpd_dict = {l[3]: l[1] for l in reader}

    names = {okpd_dict[k] for k in okpd_dict}

    names_eng = dict()
    names_eng_yandex = dict()

    try:
        names_eng = pickle.load(open("okpd_eng.pcl", 'rb'))
    except Exception as ex:
        print(ex)
    try:
        names_eng_yandex = pickle.load(open("okpd_eng_yandex.pcl", 'rb'))
    except Exception as ex:
        print(ex)

    translator = Translator()
    yandex_translator = Translater()
    yandex_translator.set_from_lang('ru')
    yandex_translator.set_to_lang('en')
    yandex_translator.set_key(
        'trnsl.1.1.20170307T175252Z.243ee3f0dfd2721a.'
        '7af76b883f12f9f4fc2d861bab7d733112aeb66d')

    texts = []
    total_len = 0

    for n_i, n in enumerate(names):
        if n in names_eng or n in names_eng_yandex:
            continue
        else:
            yandex_translator.set_text(n)
            translation = yandex_translator.translate()
            names_eng_yandex[n] = translation
            print(n, translation)
            # time.sleep(1)

    for n_i, n in enumerate(names):
        if n in names_eng:
            continue
        else:
            if total_len > 14000:
                try:
                    translations = translator.translate(texts)
                    time.sleep(2)
                    for t_i, t in texts:
                        names_eng[t] = translations[t_i]
                        print(len(names_eng), len(names))
                except Exception as ex:
                    pickle.dump(names_eng, open("okpd_eng.pcl", 'wb'))
                    translator = Translator()
                    time.sleep(1)
                texts = [n]
                total_len = len(n)
            else:
                texts.append(n)
                total_len += len(n)

    pickle.dump(names_eng, open("okpd_eng.pcl", 'wb'))
    pickle.dump(names_eng_yandex, open("okpd_eng_yandex.pcl", 'wb'))


if __name__ == "__main__":
    translate()
