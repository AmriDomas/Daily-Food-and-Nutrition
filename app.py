import pickle
import pandas as pd
from flask import Flask, request, render_template
from preprocess import preprocess_data
from sklearn.preprocessing import LabelEncoder

# Load model yang sudah dilatih
with open("xgboost_best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load Label Encoders
with open("label_encoded_data.pkl", "rb") as encoder_file:
    label_encoders = pickle.load(encoder_file)

# Inisialisasi Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        try:
            # Ambil input dari form dengan menangani error input
            data = {
                "Food_Item": request.form.get("food_item", "Eggs"),
                "Category": request.form.get("category", "Meat"),
                "Protein (g)": int(request.form.get("protein", 0)),
                "Carbohydrates (g)": int(request.form.get("carbohydrates", 0)),
                "Fat (g)": int(request.form.get("fat", 0)),
                "Fiber (g)": int(request.form.get("fiber", 0)),
                "Sugars (g)": int(request.form.get("sugars", 0)),
                "Sodium (mg)": int(request.form.get("sodium", 0)),
                "Cholesterol (mg)": float(request.form.get("cholesterol", 0.0)),
                "Meal_Type": request.form.get("meal_type", "Lunch"),
                "Water_Intake (ml)": float(request.form.get("water_intake", 0.0)),
            }

            df_food = pd.DataFrame([data])

            # ✅ **Label Encoding untuk data baru**
            categorical_cols = ["Food_Item", "Category", "Meal_Type"]
            
            label_encoders = {}

            for col in categorical_cols:
                le = LabelEncoder()
                df_food[col] = le.fit_transform(df_food[col])  # Apply encoding
                label_encoders[col] = le

            # ✅ **Pastikan fitur sesuai dengan format saat training**
            expected_features = model.feature_names_in_
            missing_features = [col for col in expected_features if col not in df_food.columns]
            extra_features = [col for col in df_food.columns if col not in expected_features]

            # Tambahkan fitur yang hilang dengan nilai default 0
            for col in missing_features:
                df_food[col] = 0

            # Hapus fitur yang tidak dikenali oleh model
            df_food = df_food[expected_features]

            # ✅ **Prediksi menggunakan model**
            prediction = model.predict(df_food)[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
