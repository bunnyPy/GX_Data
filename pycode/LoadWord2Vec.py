import gensim
from gensim.models import KeyedVectors

# Load the Word2Vec vectors (first 1 million vectors)
location = 'C:/Users/Mahendra_Jadaun/Desktop/GX/GoogleNews-vectors-negative300.bin'  # Path to downloaded file
wv = KeyedVectors.load_word2vec_format(location, binary=True, limit=1000000)

# Save vectors to flat file (CSV)
wv.save_word2vec_format('vectors.csv', binary=False)
