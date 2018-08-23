conda install pytorch torchvision -c soumith
conda install faiss-cpu -c pytorch

git clone https://github.com/facebookresearch/MUSE
python unsupervised.py --src_lang en --tgt_lang ru --src_emb doc2vec_category_vectors_maryland.txt --tgt_emb doc2vec_category_vectors_okpd.txt --n_refinement 5 --cuda False --dis_most_frequent 0