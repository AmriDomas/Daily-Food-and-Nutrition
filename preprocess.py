import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load label encoders
with open("label_encoded_data.pkl", "rb") as file:
    label_encoders = pickle.load(file)


def preprocess_data(df_food, label_encoders):
    # Hitung kolom tambahan dengan menangani pembagian nol
    df_food['Date'] = pd.to_datetime(df_food['Date'])

    df_food['Day_of_Week'] = df_food['Date'].dt.day_name()
    df_food['Weekend'] = df_food['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)
    df_food['Month'] = df_food['Date'].dt.month

    meal_counts = df_food.groupby('User_ID')['Meal_Type'].count().reset_index()
    meal_counts.rename(columns={'Meal_Type': 'Total_Meals'}, inplace=True)
    df_food = df_food.merge(meal_counts, on='User_ID', how='left')

    df_food['Protein_Percentage'] = df_food['Protein (g)'] * 4 / df_food['Calories (kcal)']
    df_food['Carb_Percentage'] = df_food['Carbohydrates (g)'] * 4 / df_food['Calories (kcal)']
    df_food['Fat_Percentage'] = df_food['Fat (g)'] * 9 / df_food['Calories (kcal)']

    food_counts = df_food.groupby(['User_ID', 'Food_Item']).size().reset_index(name='Food_Frequency')
    df_food = df_food.merge(food_counts, on=['User_ID', 'Food_Item'], how='left')

    top_category = df_food.groupby(['User_ID', 'Category']).size().reset_index(name='Category_Count')
    df_food = df_food.merge(top_category, on=['User_ID', 'Category'], how='left')

    df_food['Water_per_Calorie'] = df_food['Water_Intake (ml)'] / df_food['Calories (kcal)']

    daily_water = df_food.groupby(['User_ID', 'Date'])['Water_Intake (ml)'].sum().reset_index()
    df_food = df_food.merge(daily_water, on=['User_ID', 'Date'], how='left', suffixes=('', '_Daily'))

    df_food['Water_Diff'] = df_food['Water_Intake (ml)'] - df_food['Water_Intake (ml)_Daily']

    df_food['Water_Diff_Percentage'] = df_food['Water_Diff'] / df_food['Water_Intake (ml)_Daily']

    # Tangani pembagian dengan nol atau NaN
    df_food.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_food.fillna(0, inplace=True)

    # Label Encoding untuk kolom kategori
    categorical_cols = ["Food_Item", "Category", "Meal_Type", "Day_of_Week"]

    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_food[col] = le.fit_transform(df_food[col])  # Apply encoding
        label_encoders[col] = le

    # Hapus kolom yang tidak diperlukan
    food_clean = df_food.drop(['Date', 'User_ID'], axis=1, errors='ignore')

    return food_clean
