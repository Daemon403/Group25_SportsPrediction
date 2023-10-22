from flask import Flask,render_template, request, jsonify
import pickle,joblib
import numpy as np

app = Flask(__name__)
#model = pickle.load(open('./model.pkl', 'rb')) 
model = joblib.load(open('./modelfinal.joblib', 'rb'))

@app.route("/")
def predict():
    return render_template("index.html")


@app.route("/sub", methods = ["POST"])
def submit():
    if request.method == "POST":
        features = [int(x) for x in request.form.values()]
        features = [np.array(features)]
        #test = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        # test = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        # test = [np.array(test)]
        final = model.predict(features)
        final = round(final[0],0)
        return render_template("sub.html", fini = final)


if __name__ =="__main__":
    app.run(debug=True) 