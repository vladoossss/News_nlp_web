from nltk.corpus import stopwords
import pymorphy2
import re
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
import pickle
stop_words = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()


def text_cleaner(text, lemm=True):
    """
    Преобразование исходного текста
    :param text: входной текст
    :param lemm: лемматизация
    :return: обработанный текст
    """
    new_string = text.lower()  # lower register
    new_string = re.sub("[^а-яА-Я]", " ", new_string)  # delete all except russian
    # removing stop words
    tokens = [w for w in new_string.split() if not w in stop_words]
    # removing short words
    words = [w for w in tokens if len(w) >= 3]
    # lemmatization
    if lemm:
        words = [morph.parse(w)[0].normal_form for w in words]

    return ' '.join(words).strip()


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def model_loader(model_path):
    dependencies = {
        'get_f1': get_f1
    }
    model = load_model(model_path, custom_objects=dependencies, compile=False)
    return model


def load_all():
    # создадим словарь для хранения нужных данных и иницилизируем его главными моделями
    models = {'model_rub': model_loader('inference_files/model_rubric.h5'),
              'model_subrub': model_loader('inference_files/model_subrubric.h5')}

    # загружаем токенизатор текста новости
    with open('inference_files/tokenizer_text.pickle', 'rb') as f:
        models['tokenizer_text'] = pickle.load(f)

    # загружаем токенизаторы тегов
    with open('inference_files/binarizer_rubric.pickle', 'rb') as f:
        models['binarizer_rub'] = pickle.load(f)
    # загружаем токенизатор тегов
    with open('inference_files/binarizer_subrubric.pickle', 'rb') as f:
        models['binarizer_subrub'] = pickle.load(f)

    return models


def inference(text, model, tokenizer_text, binarizer):
    text = text_cleaner(text)

    text = tokenizer_text.texts_to_sequences([text])
    text = pad_sequences(text, 200, padding='post')

    pred = model.predict(text)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    pred = binarizer.inverse_transform(pred)
    return ' '.join(pred)
