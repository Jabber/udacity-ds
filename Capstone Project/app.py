from flask import Flask, request
from flask import render_template
from model import dog_predict

app = Flask(__name__)


@app.route('/index')
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    img_path = request.form.get('imgurl')
    prediction = dog_predict(img_path)
    return prediction


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
