# from flask import Flask, request, render_template,request, redirect
# import pandas as pd
# import joblib
# import os
# import pickle

# # Declare a Flask app
# app = Flask(__name__)

# # Main function here
# @app.route('/', methods=['GET', 'POST'])
# def main():
#     # If a form is submitted
#     if request.method == "POST":
        
#         # Unpickle classifier
#         # logreg = joblib.load("logreg.pkl")
#         # script_dir = os.path.dirname(os.path.abspath(__file__))
#         # logreg_path = os.path.join(script_dir, 'logreg.pkl')
#         # logreg = joblib.load(logreg_path)
#         with open('datmin/logreg.pkl', 'rb') as file:
#             logreg = pickle.load(file)

        
#         # Unpickle classifier
#         # knn = joblib.load("knn.pkl")
#         # script_dir = os.path.dirname(os.path.abspath(__file__))
#         # knn_path = os.path.join(script_dir, 'knn.pkl')
#         # knn = joblib.load(knn_path)
#         with open('datmin/knn.pkl', 'rb') as file:
#             knn = pickle.load(file)

#         # Get values through input bars
#         age = request.form.get("age")
#         gender = request.form.get("gender")
#         height = request.form.get("height")
#         weight = request.form.get("weight")
#         bmi = request.form.get("bmi")

#         # Put inputs to dataframe
#         X = pd.DataFrame([[1, age, gender, height, weight, bmi]], columns=["ID", "Age", "Gender", "Height", "Weight", "BMI"])
        
#         # Get prediction
#         pred_logreg = logreg.predict(X_test)[0]
#         pred_knn = knn.predict(X_test)[1]
        
#     else:
#         pred_logreg = ""
#         pred_knn = ""
        
#     return render_template("website.html", output = [pred_logreg, pred_knn])

# # Running the app
# if __name__ == '__main__':
#     app.run(debug = True)




from flask import Flask, request, render_template
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
import os
import pickle

# Mendeklarasikan aplikasi Flask
app = Flask(__name__)

# Fungsi utama di sini
@app.route('/', methods=['GET', 'POST'])
def main():
    # Jika formulir dikirimkan
    if request.method == "POST":
        # Membuka kembali model klasifikasi
        # with open('path/to/your/logreg.pkl', 'rb') as file:
        #     logreg = pickle.load(file)
        
        # logreg = pickle.load(open("D:\pawl\datmin/logreg.pkl", "rb"))
        # knn = pickle.load(open("D:\pawl\datmin/knn.pkl", "rb"))

        script_dir = os.path.dirname(os.path.abspath(__file__))
        logreg_path = os.path.join(script_dir, 'logreg.pkl')
        logreg = joblib.load(logreg_path)
        
        # Unpickle classifier
        # knn = joblib.load("knn.pkl")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        knn_path = os.path.join(script_dir, 'knn.pkl')
        knn = joblib.load(knn_path)


        # with open('path/to/your/knn.pkl', 'rb') as file:
        #     knn = pickle.load(file)

        # Mendapatkan nilai dari input formulir
        age = float(request.form.get("age"))
        gender = int(request.form.get("gender"))  # Mengasumsikan jenis kelamin adalah nilai numerik
        height = float(request.form.get("height"))
        weight = float(request.form.get("weight"))
        bmi = float(request.form.get("bmi"))

        # Menyusun input ke dalam dataframe
        X = pd.DataFrame([[age, gender, height, weight, bmi]], columns=["Age", "Gender", "Height", "Weight", "BMI"])

        # Melakukan prediksi
        pred_logreg = logreg.predict(X)[0]
        pred_knn = knn.predict(X)[0]  # Mengasumsikan knn.predict mengembalikan array

    else:
        pred_logreg = ""
        pred_knn = ""

    return render_template("website.html", output=[pred_logreg, pred_knn])

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
