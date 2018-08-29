conda install pytorch torchvision -c soumith
conda install faiss-cpu -c pytorch

git clone https://github.com/facebookresearch/MUSE
python unsupervised.py --src_emb doc2vec_category_vectors_maryland.txt --tgt_emb doc2vec_category_vectors_okpd.txt --n_refinement 5 --cuda False --dis_most_frequent 0 --epoch_size 5000 --n_epochs 1000 --dico_train default --dico_build S2T --dico_method csls_knn_10 --dico_max_rank 4000 --src_lang maryland --tgt_lang okpd --dico_eval maryland2okpd_dict.txt

python supervised.py --src_emb doc2vec_category_vectors_maryland.txt --tgt_emb doc2vec_category_vectors_okpd.txt --cuda False --max_vocab -1 --src_lang maryland --tgt_lang okpd --dico_train maryland2okpd_dict.txt


python3 map_embeddings.py --unsupervised                                doc2vec_category_vectors_maryland.txt doc2vec_category_vectors_okpd.txt vectors-maryland.txt      vectors-okpd.txt
python3 map_embeddings.py --semi_supervised okpd-maryland.5000-6500.txt doc2vec_category_vectors_maryland.txt doc2vec_category_vectors_okpd.txt vectors-maryland_semi.txt vectors-okpd_semi.txt

python3 map_embeddings.py --unsupervised vectors-maryland-averaged.txt vectors-okpd-averaged.txt vectors-maryland_av.txt vectors-okpd_av.txt
