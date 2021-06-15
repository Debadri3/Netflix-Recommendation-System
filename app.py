# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the XGB regressor model
filename = 'netflix_xgb1.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    
    if request.method == 'POST':
        
     data1 = float(request.form['glob_avg'])
     data2 = float(request.form['sur1'])
     data3 = float(request.form['sur2'])
     data4 = float(request.form['sur3'])
     data5 = float(request.form['sur4'])
     data6 = float(request.form['sur5'])
     data7 = float(request.form['smr1'])
     data8=  float(request.form['smr2'])
     data9=  float(request.form['smr3'])
     data10=  float(request.form['smr4'])
     data11=  float(request.form['smr5'])
     data12=  float(request.form['u_avg'])
     data13=  float(request.form['m_avg'])
     final_features = np.array([[data10,data2,data9,data7,data12,data5,data6,data1,data11,data13,data3,data8,data4]])
     
     prediction = regressor.predict(final_features)
            
            
        
              
    return render_template('index.html', prediction_text='The rating for the user-movie pair should be around {}'.format(prediction))



if __name__ == '__main__':
	app.run(debug=True)