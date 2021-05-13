from predict_tags import text_cleaner
import razdel
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def load_word2vec():
    with open('inference_files/word2vec.pickle', 'rb') as f:
        model_w2v = pickle.load(f)
    return model_w2v


def textrank(text, model_w2v):
    # разбиваем текст на предложения
    sentences = [sentence.text for sentence in razdel.sentenize(text)]

    # очищаем предложения
    clean_sentences = [text_cleaner(sen) for sen in sentences]

    # разбиваем предложения на слова
    sentence_words = [[token.text for token in razdel.tokenize(sentence)] for sentence in clean_sentences]

    # вычисляем вектора для каждого предложения в тексте
    sentence_vectors = []
    for words in sentence_words:
        v = np.zeros((100,))
        if len(words) != 0:
            for w in words:
                try:
                    v += model_w2v[w]
                except:
                    v += np.zeros((100,))

            v = v / len(words)
            sentence_vectors.append(v)
        else:
            sentence_vectors.append(v)

    # матрица похожести
    sim_mat = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = \
                    cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

    # строим граф, считаем пейджранки и сортируем по ним
    graph = nx.from_numpy_array(sim_mat)
    pr = nx.pagerank_numpy(graph)

    return sorted(((i, pr[i], s) for i, s in enumerate(sentences) if i in pr),
                  key=lambda x: pr[x[0]], reverse=True)


# выводим самое популярное предложение для заголовка
def extract_summary(text, model_w2v, n=2):
    tr = textrank(text, model_w2v)
    top_n = sorted(tr[:n])
    return ' '.join(x[2] for x in top_n)
