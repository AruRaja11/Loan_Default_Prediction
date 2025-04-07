# 📊 Loan Default Prediction Using Machine Learning
## ✅ Objective:
The objective of this project is to build a machine learning model that can predict whether a person is eligible for a loan or not, based on their financial and demographic data. The final application helps financial institutions make informed lending decisions and reduce the risk of loan defaults.
# 🗂️ Project Structure:
* loan_data.csv
  Contains the historical data of loan applicants including features like income, credit score, employment status, etc.
* Data Processing.ipynb
  * Cleaned the raw dataset by handling missing values, outliers, and inconsistent formats.
  * Encoded categorical variables and normalized numerical values.
  * Ensured data was suitable for machine learning models.
* Exploratory Data Analytics.ipynb
  * Conducted in-depth analysis and visualizations to understand feature distributions and correlations.
  * Plotted histograms, boxplots, heatmaps, and count plots.
  * Identified key trends such as the impact of credit score and employment status on loan approval.
* Model building.ipynb
  * Tried multiple classification algorithms including:
    * Logistic Regression
    * Random Forest Classifier
    * Decision Tree Classifier
  * Evaluated models using accuracy, precision, recall, and F1-score.
  * Applied GridSearchCV to tune hyperparameters of the Decision Tree.
  * Final model selected: Decision Tree Classifier with tuned parameters, achieving high prediction accuracy.
* app.py
  * Developed a Streamlit web application to make predictions based on user inputs.
  * Users can enter custom values through a Sidebar UI (e.g., income, credit score, loan amount).
  * Model processes the input and displays the predicted result: Loan Approved / Not Approved.
  * Clean and interactive interface for demonstration.
# 📌 Technologies Used:
* Python
* Pandas, NumPy – Data handling
* Matplotlib, Seaborn – Visualization
* Scikit-learn – Machine Learning
* Streamlit – Web application
# ✅ Requirements:
1. Install Steamlit
2. Install Annotated_text
# 🚀 How to Run:
1. Execute all Three files
2. Run app.py
# ✅ Results:
* Final model: Decision Tree Classifier with tuned hyperparameters.
* Achieved significant accuracy and precision in predicting loan defaults.
* Streamlit app provides a user-friendly interface for practical use.
