from gensim.models import KeyedVectors
from gensim.utils import tokenize

word_vectors = KeyedVectors.load("SG-300-W10N20E50/W2V.kv", mmap='r+')

def embedding(vec):
    sentencesMatrix = []

    for row in vec:
        tokens = tokenize(text=row,lowercase=True)
        sentence = []
        for tok in tokens:
            if tok in word_vectors:
                word = word_vectors[tok]
            else:
                word = [0 for _ in range(300)]
            sentence.append(word)
        if sentence:
            sentencesMatrix.append(sentence)
    return sentencesMatrix