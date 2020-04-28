from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
from sklearn import linear_model
from pandas import DataFrame
stocks = pd.read_csv("college_predictor.csv")
df = DataFrame(stocks, columns=['GRE Score', 'TOEFL Score', 'CGPA', 'Chance of Admit '])
X = df[['GRE Score', 'TOEFL Score', 'CGPA']]
Y = df['Chance of Admit ']
regr = linear_model.LinearRegression()
regr.fit(X, Y)
f=regr.score(X,Y)
print("Accuracy is ",f)
regr.coef_
print(regr.coef_)
regr.intercept_
print(regr.intercept_)

app = Flask(__name__)



@app.route('/')
def welcome():
    return render_template('input.html')


@app.route('/scores', methods=['POST', 'GET'])
def scores():
    name = request.form["name"]
    gre = request.form["gre"]
    toefl = request.form["toefl"]
    cgpa = request.form["cgpa"]
    gre = int(gre)
    toefl = int(toefl)
    cgpa = float(cgpa)
    admit_chance = regr.predict([[gre, toefl, cgpa]])
    admit_chance = admit_chance[0]
    admit_chance = round(admit_chance*100, 2)
    return render_template('output.html', name=[ name ,gre,toefl,cgpa, admit_chance])




if __name__ == "__main__":
    app.run(debug=True)