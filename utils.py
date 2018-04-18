import os
import tarfile
# NLP
import gensim

WORD2VEC_WEIGHTS = 'GoogleNews-vectors-negative300-top100000.bin'
if WORD2VEC_WEIGHTS not in os.listdir():
    WORD_LIMIT = 100000
    WORD2VEC_TAR = 'GoogleNews-vectors-negative300-top100000.tar.gz'

    # extract word2vec zipped file (circumvent github max file size)
    tar = tarfile.open(WORD2VEC_TAR, "r:gz")
    tar.extractall()
    tar.close()
    print('finished extracting')

# init word2vec model, limit to common words
model = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_WEIGHTS, binary=True, limit=WORD_LIMIT)
# can save top X most common words in bin format
# gensim.models.KeyedVectors.save_word2vec_format(model, fname='reduced.bin', binary=True)
