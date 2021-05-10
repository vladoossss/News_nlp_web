from nltk.corpus import stopwords
import pymorphy2
import re
import nltk
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
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
    model = load_model(model_path, custom_objects=dependencies)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=[get_f1])

    return model


def inference(text, model_path, tokenizer_path, binarizer_path):
    text = text_cleaner(text)

    # загружаем токенизатор текста новости
    with open(tokenizer_path, 'rb') as f:
        tokenizer_text = pickle.load(f)

    text = tokenizer_text.texts_to_sequences([text])
    text = pad_sequences(text, 200, padding='post')

    # загружаем модель
    model = model_loader(model_path)
    pred = model.predict(text)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    # загружаем токенизатор тегов
    with open(binarizer_path, 'rb') as f:
        binarizer = pickle.load(f)

    pred = binarizer.inverse_transform(pred)
    return ' '.join(pred)
