from gensim.models import KeyedVectors
import os

# Function to convert GloVe format to Word2Vec format
def convert_glove_to_word2vec(glove_file, word2vec_file):
    """
    Convert GloVe file to Word2Vec format.
    :param glove_file: Path to the GloVe file.
    :param word2vec_file: Path to save the Word2Vec formatted file.
    """
    # Load the GloVe embeddings directly using KeyedVectors.load_word2vec_format
    # and then save it in the desired format.
    model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True) # The glove file doesn't have a header
    model.save_word2vec_format(word2vec_file, binary=False)

    return model

def load_glove_embeddings(glove_file, word2vec_file):

    print(f"* Loading glove embeddings from file: {glove_file}")

    # Convert GloVe to Word2Vec format
    if not os.path.exists(word2vec_file):
        convert_glove_to_word2vec(glove_file, word2vec_file)
    else:
        print(f"word2vec_file already exists: {word2vec_file}")

    # Load the Word2Vec formatted embeddings
    word_vectors_glove = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)

    return word_vectors_glove