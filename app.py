from flask import Flask,render_template,url_for,request
import pandas as pd
import model2

app = Flask(__name__)

@app.route("/")
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        post_title = request.form['post_title']
        post_body = request.form['post_body']
        post = post_title + '. ' + post_body
        post = [str(post)]
        prediction = model2.predict(post)
    return render_template('result.html', result = prediction)


if __name__ == '__main__':
	app.run(debug=True)
