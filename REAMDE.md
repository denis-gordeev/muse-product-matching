To get translations for OKPD2-codes you need to launch translate_okpd.py (it translates OKPD2-descriptions using either Yandex or Google-api)

To get doc2vec embeddings you need to launch train_embeddings.py; you need to specify your folder for russian_fasttext and english_fasttext and preprocess your taxonomy files in a correct order 'okpds3_unique' and 'eMaryland_Marketplace_Bids_-_Fiscal_Year_{}.csv' are used in my case and are processed to output 'x' as text and 'y' as the category.

non_muse.py is a string matching dataset


vecmap launch example

~~~
python unsupervised.py --src_lang en --tgt_lang ru --src_emb doc2vec_category_vectors_maryland.txt --tgt_emb doc2vec_category_vectors_okpd.txt --n_refinement 5 --cuda False --dis_most_frequent 0
~~~

MAX Gensim version is 3.2.0 (otherwise Doc2Vec can't use pre-trained word2vec embeddings)


please contact me if any questions arise