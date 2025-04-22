#import required packages
import pandas as pd
import streamlit as st
import pickle
from annotated_text import annotated_text

# Read datasets
raw = pd.read_csv('imp_raw_data.csv')
data = pd.read_csv('important_data.csv')
data = data.iloc[:, 1:]
raw = raw[data.columns]
data.dropna(inplace=True)

data.drop('TARGET', axis=1, inplace=True)
raw.drop('TARGET', axis=1, inplace=True)

# Fixed Header
st.markdown("""
    <style>
    .fixed-header {
        max-height:250px;
        width: 100%;
        background-color: #AB29F7; 
        color: white; 
        text-align: center;
        font-size: 20px;
        border-bottom: 2px solid #ccc;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .fixed-header img {
        height: 90px;
        width: 90px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="fixed-header">
    <img src="https://png.pngtree.com/png-vector/20190121/ourmid/pngtree-vector-loan-icon-png-image_334585.jpg" >
    <h1>Loan Default Prediction</h1>
    </div>
    """, unsafe_allow_html=True)

for i in range(2):
    st.write("  ")


st.markdown('<div style="background-color: #AB29F7; padding: 5px; border-radius: 8px; display: inline-block;"><h6>Use Filters Tab For Feature Selection!</h6></div>', unsafe_allow_html=True)


# Define categorical and numerical columns separately
categorical_columns = []
numerical_columns = []

for i in raw.columns:
    if (raw[i].dtype == 'object' or raw[i].nunique() <=10):
        categorical_columns.append(i)
    else:
        numerical_columns.append(i)

# Sidebar for filters
st.sidebar.header('Feature Selection')
with st.sidebar.form('Filters and Options'):

    # Dynamic selection for categorical columns
    categorical_values = {}
    for col in categorical_columns:
        unique_values = raw[col].dropna().unique().tolist()
        try:
            categorical_values[col] = st.selectbox(f"{col.replace('_', '')}", options=unique_values)
        except: 
            st.write(col)

    # Dynamic sliders for numerical columns
    numerical_values = {}
    for col in numerical_columns:
        min_val = raw[col].min()
        max_val = raw[col].max()
        default_val = raw[col].median()
        try:
            if pd.api.types.is_integer_dtype(raw[col]):
                numerical_values[col] = st.slider(f"{col.replace('_', '')}", int(min_val), int(max_val), int(default_val))
            else:
                numerical_values[col] = st.slider(f"{col.replace('_', '')}", float(min_val), float(max_val), float(default_val))

        except:
            st.write(col)
    submitted = st.form_submit_button('predict')


# Combine all selected features into a list
features = [
    numerical_values.get(col, None) if col in numerical_columns else categorical_values.get(col, None)
    for col in (numerical_columns + categorical_columns)
]

# Convert categorical values into their index representation (optional)
for col in categorical_columns:
    if col in categorical_values:
        unique_values = list(raw[col].dropna().unique())
        encoded_value = unique_values.index(categorical_values[col])
        idx = (numerical_columns + categorical_columns).index(col)
        features[idx] = encoded_value

# Annotate text 
st.write(" ")
annotated_items =[(str(key), str(value)) for key, value in categorical_values.items()]
annotated_text(*annotated_items)

#annotated_items = [(str(key), str(value)) for key, value in numerical_values.items()]
#annotated_text(annotated_items)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

if submitted:
    prediction = round(model.predict([features])[0], 2)
    if prediction == 0:
        result = "Not Eligible for Loan"
    else:
        result = "Eligible for Loan"
    st.markdown(f"<h2>The expected output: {result}</h2>", unsafe_allow_html=True)
else:
    st.markdown(f"<h4>Adjust the sidebar and click Predict to see the prediction</h4>", unsafe_allow_html=True)

st.markdown(f'<p>Model is 94% accurate</p>', unsafe_allow_html=True)

st.write('___')