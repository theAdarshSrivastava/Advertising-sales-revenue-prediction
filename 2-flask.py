from flask import Flask,render_template,request
import numpy as np
import joblib

model=joblib.load('Advertising_sales_revenue.sav')

app=Flask(__name__)

@app.route('/')
def page():
	return render_template('home.html')

@app.route('/getresults',methods=['POST'])
def getresults():
	form_data=request.form

	tv=float(form_data['tv'])
	radio=float(form_data['radio'])
	newspaper=float(form_data['newspaper'])

	test_data=np.array([tv,radio,newspaper]).reshape(1,3)

	prediction=model.predict(test_data)[0][0]

	result_dict={"revenue":round(prediction,3)}

	return render_template('results.html',results=result_dict)



app.run(debug=True)
