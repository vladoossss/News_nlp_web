from flask import Flask, render_template, request
from predict_tags import load_all, inference
from summary import load_word2vec, extract_summary


# Создаем объект app класса Flask, который будет обращаться к нашему файлу
app = Flask(__name__)


# Главная страница
@app.route('/')
def index():
    return render_template('start.html')


@app.route('/tags', methods=['POST', 'GET'])
def predict_tags():
    models = load_all()

    if request.method == 'GET':
        return render_template('predict_tags.html',
                               text_in='',
                               text_out=''
                               )

    if request.method == 'POST':
        result = request.form
        text_in = result['text_in']

        result_1 = inference(text_in, models['model_rub'],
                             models['tokenizer_text'], models['binarizer_rub'])
        result_2 = inference(text_in, models['model_subrub'],
                             models['tokenizer_text'], models['binarizer_subrub'])
        res = '#' + result_1 + ', ' + '#' + result_2

        return render_template('predict_tags.html',
                               test=True,
                               text_in=text_in,
                               text_out=res
                               )


@app.route('/summarization', methods=['POST', 'GET'])
def summary():
    model_w2v = load_word2vec()

    if request.method == 'GET':
        return render_template('summary.html',
                               text_in='',
                               count='',
                               text_out=''
                               )

    if request.method == 'POST':
        result = request.form

        text_in = result['text_in']
        count = result['count']
        if not count:
            text_out = extract_summary(text_in, model_w2v)

            return render_template('summary.html',
                                   test=True,
                                   text_in=text_in,
                                   text_out=text_out
                                   )

        else:
            count = int(count)
            text_out = extract_summary(text_in, model_w2v, count)

            return render_template('summary.html',
                                   test=True,
                                   text_in=text_in,
                                   count=count,
                                   text_out=text_out
                                   )


# Запускаем сервер
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=1234) #, host='0.0.0.0')
