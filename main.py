from flask import Flask, render_template, request
from predict_tags import inference
from summary import extract_summary


# Создаем объект app класса Flask, который будет обращаться к нашему файлу
app = Flask(__name__)


# Главная страница
@app.route('/')
def index():
    return render_template('start.html')


@app.route('/tags', methods=['POST', 'GET'])
def predict_tags():
    if request.method == 'GET':
        return render_template('predict_tags.html',
                               text_in='',
                               text_out=''
                               )

    else:
        result = request.form
        text_in = result['text_in']

        result_1 = inference(text_in, 'models/model_rubric.h5',
                             'models/tokenizer_text.pickle', 'models/binarizer_rubric.pickle')
        result_2 = inference(text_in, 'models/model_subrubric.h5',
                             'models/tokenizer_text.pickle', 'models/binarizer_subrubric.pickle')
        res = '#' + result_1 + ', ' + '#' + result_2

        return render_template('predict_tags.html',
                               test=True,
                               text_in=text_in,
                               text_out=res
                               )


@app.route('/summarization', methods=['POST', 'GET'])
def summary():
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
            text_out = extract_summary(text_in)

            return render_template('summary.html',
                                   test=True,
                                   text_in=text_in,
                                   text_out=text_out
                                   )

        else:
            count = int(count)
            text_out = extract_summary(text_in, count)

            return render_template('summary.html',
                                   test=True,
                                   text_in=text_in,
                                   count=count,
                                   text_out=text_out
                                   )


# Запускаем сервер
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=1234)
