#import required packages
import pandas as pd
import streamlit as st
import pickle
from annotated_text import annotated_text
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

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
        background-color: #ff4b4b; 
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
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRGVHlKC2_1eCxt6K1DVxMM6e6jNAI2CmwbVw&s" >
    <h1>Loan Default Prediction</h1>
    </div>
    """, unsafe_allow_html=True)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    
page_select = option_menu(None, ['Home', 'Upload', 'Info'], icons=['house', 'cloud-upload', 'gear'], menu_icon='cast', orientation='horizontal')


if page_select == 'Home':
    
    for i in range(2):
        st.write("  ")
    
    
    st.markdown('<div style="background-color: #ff4b4b; padding: 5px; border-radius: 8px; display: inline-block;"><h5>Use Filters Tab For Feature Selection!</h5></div>', unsafe_allow_html=True)
    
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
    

    
    if submitted:
        prediction = round(model.predict([features])[0], 2)
        if prediction == 0:
            result = "Eligible for Loan"
        else:
            result = "Not Eligible for Loan"
        st.markdown(f"<h2>The expected output: {result}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h4>Adjust the sidebar and click Predict to see the prediction</h4>", unsafe_allow_html=True)
    
    st.markdown(f'<p>Model is 85% accurate</p>', unsafe_allow_html=True)
    
    st.write('___')

elif page_select == 'Upload':
    try:
        st.markdown('<div style="background-color: #ff4b4b; padding: 5px; border-radius: 8px; display: inline-block;"><h5>Upload your dataset for batch predictions!</h5></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader('Choose a File')
    
        if uploaded_file is not None:
            def pass_text_value(row):
                if row == 0:
                    return 'Eligible'
                else:
                    return 'Not Eligible'
            def clean_data(dataframe):
                categorical = [i for i in dataframe.columns if dataframe[i].dtype in ['str', 'object']]
                numerical = [i for i in dataframe.columns if dataframe[i].nunique() > 10]
    
                for i in categorical:
                    dataframe[i] =dataframe[i].fillna(dataframe[i].mode()[0])
                    dataframe[i] = label.fit_transform(dataframe[i])
                for j in numerical:
                    dataframe[j] = dataframe[j].fillna(dataframe[j].mean())
                    if dataframe[j].iloc[0] < 0:
                        dataframe[j] = dataframe[j].abs()
                    else:continue
                return dataframe
                    
            
            input = pd.read_csv(uploaded_file)
            user_data = clean_data(input)
            user_data = user_data.iloc[:, 1:]
            predictions = pd.DataFrame(model.predict(user_data),columns=['TARGET PREDICTED'])
            result_df = pd.concat([user_data, predictions], axis=1)
            result_df['TARGET PREDICTED'] = result_df['TARGET PREDICTED'].apply(pass_text_value)
            st.write(result_df)
    except:
        st.write('Oops! some error occured.')
    st.write('---')

        
elif page_select == 'Info':
    st.markdown('''<h4 style="color:#ff4b4b">🔷 Home Page Description</h4>''', unsafe_allow_html = True)
    st.write('On the Home Page of your Streamlit app, users can interactively predict loan default by modifying 15 key input features using sliders and selection boxes. After adjusting the inputs, they can click a "Predict" button to get a prediction — indicating whether the applicant is likely to default or not default on a loan. This feature makes your model easily accessible and user-friendly for both technical and non-technical users.')
    st.markdown('''<dl>
  <dt style="color:#ff4b4b">EXT_SOURCE_3</dt>
  <dd>A normalized external score from a third-party credit bureau or model. Higher values indicate better creditworthiness.</dd>
  <dt style="color:#ff4b4b">EXT_SOURCE_2</dt>
  <dd>Another external source score reflecting the applicant’s credit standing. Also positively correlated with loan repayment likelihood.</dd>
    <dt style="color:#ff4b4b">DAYS_EMPLOYED</dt>
  <dd>Number of days the applicant has been employed (negative values; more negative = longer employment).</dd>
  <dt style="color:#ff4b4b">DAYS_BIRTH</dt>
  <dd>Age of the applicant in days (negative; convert to positive to get actual age).</dd>
  <dt style="color:#ff4b4b">DAYS_LAST_PHONE_CHANGE</dt>
  <dd>Days since the last change of phone number. May indicate stability.</dd>
  <dt style="color:#ff4b4b">DAYS_REGISTRATION</dt>
  <dd>Days since the client first registered in the system (also negative).</dd>
    <dt style="color:#ff4b4b">AMT_CREDIT_x</dt>
  <dd>Total amount of credit applied for the loan.</dd>
  <dt style="color:#ff4b4b">DAYS_ID_PUBLISH</dt>
  <dd>Days since the applicant’s ID was published/issued.</dd>
  <dt style="color:#ff4b4b">AMT_ANNUITY_x</dt>
  <dd>Annual loan annuity — the expected yearly payment.</dd>
  <dt style="color:#ff4b4b">AMT_GOODS_PRICE_x</dt>
  <dd>Price of the goods for which the loan is taken (e.g., car, house).</dd>
  <dt style="color:#ff4b4b">SK_ID_CURR</dt>
  <dd>Unique identifier for each loan applicant. Not useful for prediction but good for tracking.</dd>
  <dt style="color:#ff4b4b">ORGANIZATION_TYPE</dt>
  <dd>The applicant's employer category (e.g., Business Entity, Government, School).</dd>
  <dt style="color:#ff4b4b">REGION_POPULATION_RELATIVE</dt>
  <dd>Population density of the region the applicant lives in.</dd>
  <dt style="color:#ff4b4b">DAYS_DECISION</dt>
  <dd>Days since the loan decision was made.</dd>
  <dt style="color:#ff4b4b">AMT_INCOME_TOTAL</dt>
  <dd>Applicant's total income, often used to calculate credit-to-income ratios.</dd>
</dl>''', unsafe_allow_html=True)
    st.markdown('''<h4 style="color:#ff4b4b">🔷 About Upload Page</h4>''', unsafe_allow_html = True)
    st.write('On the second page of your Streamlit app, users can perform batch predictions by uploading a CSV file containing multiple rows of input data. Each row represents a unique loan applicant with the same 15 features used in the single prediction mode. Once uploaded, the app processes the DataFrame and returns predictions for all entries — showing whether each applicant is likely to default or not default on their loan.')
    st.markdown('''The featuer especially useful for:''', unsafe_allow_html=True)
    st.markdown('''<ul><li>Financial analysts processing large volumes of applications.</li><li>Organizations looking to assess loan eligibility in bulk.</li><li>Testing the model with real-world datasets.</li></ul>''', unsafe_allow_html=True)
