from flask import Flask,request,jsonify,render_template
import numpy as np
from keras.models import load_model


app=Flask("__name__",template_folder="template")
#model=pickle.load(open('ANN_Regression.pkl','rb'))
mymodel=load_model('ANN_Regression.h5')


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final_features=np.array([int_features])
    final_features.reshape(1,8)

    prediction=mymodel.predict(final_features)
    print(prediction)

    #output=np.round(prediction[0],2)
    return render_template('result.html', prediction=prediction)

    #return render_template('index.html',prediction_text='PM 2.5 value should be${}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
