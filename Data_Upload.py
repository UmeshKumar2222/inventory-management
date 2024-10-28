import pandas as pd
import numpy as np
import streamlit as st
import logging

# Configure the Streamlit page
st.set_page_config(page_title="Inventory Management", layout='wide')

# tab1 = st.tabs(["Upload Dataset"])
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


st.title("Data Upload")

def imputing_na(data, column, method):
    if method == 'mean':
        data[column] = data[column].fillna(data[column].mean())
    elif method == 'median':
        data[column] = data[column].fillna(data[column].median())
    elif method == 'mode':
        mode_value = data[column].mode()[0]  # Take the first mode value if multiple modes
        data[column] = data[column].fillna(mode_value)
    return data

def load_data(uploaded_file):
    """Load CSV data with error handling."""
    try:
        return pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")
        return None

def main():
    # File uploaders for different datasets
    uploaded_file1 = st.file_uploader("Upload beginning inventory dataset here", type="csv")
    uploaded_file2 = st.file_uploader("Upload end inventory dataset here", type="csv")
    uploaded_file3 = st.file_uploader("Upload purchase final dataset here", type="csv")
    uploaded_file4 = st.file_uploader("Upload sales dataset here", type="csv")
    uploaded_file5 = st.file_uploader("Upload Customer dataset here", type="csv")

    # Load datasets into session state
    if uploaded_file1:
        if "data1" not in st.session_state:
            st.session_state.data1 = load_data(uploaded_file1)

    if uploaded_file2:
        if "data2" not in st.session_state:
            st.session_state.data2 = load_data(uploaded_file2)

    if uploaded_file3:
        if "data3" not in st.session_state:
            st.session_state.data3 = load_data(uploaded_file3)

    if uploaded_file4:
        if "data4" not in st.session_state:
            st.session_state.data4 = load_data(uploaded_file4)
    
    if uploaded_file5:
        if "data5" not in st.session_state:
            st.session_state.data5 = load_data(uploaded_file5)

    # Retrieve datasets from session state with error handling
    data1 = st.session_state.get('data1', None)
    data2 = st.session_state.get('data2', None)
    data3 = st.session_state.get('data3', None)
    data4 = st.session_state.get('data4', None)
    data5 = st.session_state.get('data5', None)

    # Display information for each dataset
    datasets = {
        "Beginning Inventory": data1,
        "End Inventory": data2,
        "Purchase Final": data3,
        "Sales": data4,
        "Customer": data5
    }

    for title, data in datasets.items():
        if data is not None:
            st.write(f"Data Shape for {title}:", data.shape)
            st.write(f"First few rows of the data for {title}:")
            st.dataframe(data.head())
        else:
            st.warning(f"{title} data is not uploaded yet.")
#.................................................................................
    if (data1 is not None): 
        if st.session_state.data1.isna().any().any():
                columns_with_nulls = st.session_state.data1.columns[st.session_state.data1.isnull().sum() > 0].tolist()
                null_col = st.selectbox("Select column to remove the null value in Beginning Inventory Dataset", columns_with_nulls)
                
                if st.session_state.data1[null_col].dtype == 'O':
                    method = st.selectbox(f"Select method to fill NA in {null_col}", ['Select method', 'mode'], index=1)
                else:
                    method = st.selectbox(f"Select method to fill NA in {null_col}", ['Select method', 'mean', 'median', 'mode'], index=1)
                
                if method != 'Select method' and method:
                    if st.button("Impute Beginning Inventory"):
                        st.session_state.data1 = imputing_na(st.session_state.data1, null_col, method)
                        st.success(f"Imputed column '{null_col}' using '{method}' method.")
                        logger.info(f"Imputed column '{null_col}' using '{method}' method.")
    
    if (data2 is not None): 
        if st.session_state.data2.isna().any().any():
            columns_with_nulls = st.session_state.data2.columns[st.session_state.data2.isnull().sum() > 0].tolist()
            null_col = st.selectbox("Select column to remove the null value in End Inventory Dataset", columns_with_nulls, index=0)
            
            if st.session_state.data2[null_col].dtype == 'O':
                method = st.selectbox(f"Select method to fill NA in {null_col}", ['Select method', 'mode'], index=1)
            else:
                method = st.selectbox(f"Select method to fill NA in {null_col}", ['Select method', 'mean', 'median', 'mode'], index=1)
            
            if method != 'Select method' and method:
                if st.button("Impute End Inventory"):
                    st.session_state.data2 = imputing_na(st.session_state.data2, null_col, method)
                    st.success(f"Imputed column '{null_col}' using '{method}' method")
    
    if (data3 is not None): 
        if st.session_state.data3.isna().any().any():
            columns_with_nulls = st.session_state.data3.columns[st.session_state.data3.isnull().sum() > 0].tolist()
            null_col = st.selectbox("Select column to remove the null value in Purchase Final Dataset", columns_with_nulls)
            
            if st.session_state.data3[null_col].dtype == 'O':
                method = st.selectbox(f"Select method to fill NA in {null_col}", ['Select method', 'mode'], index=1)
            else:
                method = st.selectbox(f"Select method to fill NA in {null_col}", ['Select method', 'mean', 'median', 'mode'], index=1)
            
            if method != 'Select method' and method:
                if st.button("Impute purschase dataset"):
                    st.session_state.data3 = imputing_na(st.session_state.data3, null_col, method)
                    st.success(f"Imputed column '{null_col}' using '{method}' method")

    if (data4 is not None): 
        if st.session_state.data4.isna().any().any():
            columns_with_nulls = st.session_state.data4.columns[st.session_state.data4.isnull().sum() > 0].tolist()
            null_col = st.selectbox("Select column to remove the null value in Sales Dataset in Sales Dataset", columns_with_nulls)
            
            if st.session_state.data4[null_col].dtype == 'O':
                method = st.selectbox(f"Select method to fill NA in {null_col}", ['Select method', 'mode'], index=1)
            else:
                method = st.selectbox(f"Select method to fill NA in {null_col}", ['Select method', 'mean', 'median', 'mode'], index=1)
            
            if method != 'Select method' and method:
                if st.button("Impute sales dataset"):
                    st.session_state.data4 = imputing_na(st.session_state.data4, null_col, method)
                    st.success(f"Imputed column '{null_col}' using '{method}' method")
    
    if (data5 is not None): 
        if st.session_state.data5.isna().any().any():
            columns_with_nulls = st.session_state.data5.columns[st.session_state.data5.isnull().sum() > 0].tolist()
            null_col = st.selectbox("Select column to remove the null value in Sales Dataset in Sales Dataset", columns_with_nulls)
            
            if st.session_state.data5[null_col].dtype == 'O':
                method = st.selectbox(f"Select method to fill NA in {null_col}", ['Select method', 'mode'], index=1)
            else:
                method = st.selectbox(f"Select method to fill NA in {null_col}", ['Select method', 'mean', 'median', 'mode'], index=1)
            
            if method != 'Select method' and method:
                if st.button("Impute sales dataset"):
                    st.session_state.data5 = imputing_na(st.session_state.data5, null_col, method)
                    st.success(f"Imputed column '{null_col}' using '{method}' method")


# Run the app
if __name__ == "__main__":
    main()
