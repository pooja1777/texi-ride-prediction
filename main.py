from flask import Flask ,request ,render_template ,jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features  = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output=int(prediction)
    return render_template('index.html', prediction_text="Predicted rides are: {}".format(output))
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000)
