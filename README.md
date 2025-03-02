# Daily Food and Nutrition Calories Prediction

## Project Overview
This project aims to predict the **calories (kcal)** in food consumption using **XGBoost Regression** with **hyperparameter tuning via GridSearchCV**. The dataset consists of **10,000 food consumption records**, including various nutritional information and meal types.

## Dataset Description
The dataset contains 14 columns:

| Column Name         | Description                            | Data Type |
|---------------------|----------------------------------------|-----------|
| Date               | Date of food consumption               | Object    |
| User_ID            | Unique identifier for users           | Int64     |
| Food_Item          | Name of the consumed food             | Object    |
| Category           | Food category (e.g., fruits, dairy)   | Object    |
| Calories (kcal)    | Target variable (calories per item)   | Int64     |
| Protein (g)        | Protein content in grams              | Float64   |
| Carbohydrates (g)  | Carbohydrates content in grams        | Float64   |
| Fat (g)            | Fat content in grams                  | Float64   |
| Fiber (g)          | Fiber content in grams                | Float64   |
| Sugars (g)         | Sugar content in grams                | Float64   |
| Sodium (mg)        | Sodium content in milligrams          | Int64     |
| Cholesterol (mg)   | Cholesterol content in milligrams     | Int64     |
| Meal_Type          | Type of meal (e.g., breakfast, lunch) | Object    |
| Water_Intake (ml)  | Water intake in milliliters           | Int64     |

## Machine Learning Model
We use **XGBoost Regression** as the predictive model with **GridSearchCV** for hyperparameter tuning. The model takes nutritional values and food categories as input features and predicts the calorie content.

### Steps Involved:
1. **Data Preprocessing**
   - Handle missing values (if any)
   - Encode categorical features
   - Normalize numerical features (if required)

2. **Feature Engineering**
   - Select relevant features
   - Perform exploratory data analysis (EDA)

3. **Model Training**
   - Train an XGBoost Regression model
   - Tune hyperparameters using GridSearchCV
   - Evaluate performance using metrics MAPE

4. **Prediction & Deployment**
   - Test the model on new data
   - Deploy using a flask

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/AmriDomas/Daily-Food-and-Nutrition.git
```
2. Navigate to the project directory:
```bash
cd daily-food-calories-prediction
```
3. Run the model training script:
```bash
python app.py
```

## Expected Output
The trained XGBoost model will predict calorie intake based on the nutritional composition of food items. The performance metrics and feature importance will be displayed after training.

## Future Enhancements
- Improve feature selection techniques
- Implement deep learning approaches for better accuracy
- Develop a web-based interface for real-time calorie prediction

## Author
[Muh Amri Sidiq] [https://www.linkedin.com/in/muh-amri-sidiq/]


