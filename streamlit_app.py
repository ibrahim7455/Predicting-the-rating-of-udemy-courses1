import streamlit as st 
import pandas as pd
import gzip
import pickle
from data_preprocessing import load_and_preprocess_data
from model import train_model

# Function to save the model as .pkl.gz
def save_compressed_model(model, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(model, f)

# Function to load the model from .pkl.gz
def load_compressed_model(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

def load_data(): 
    df, X, y, le_title, le_wishlist, scaler = load_and_preprocess_data(r'C:\Users\Test\Downloads\Udemy project\dataset\Cleaned_Udemy_data.csv')
    return df, X, y, le_title, le_wishlist, scaler

def main():
    st.title('Udemy Course Rating Prediction')

    # Load and preprocess data
    df, X, y, le_title, le_wishlist, scaler = load_data()

    # Train the model and save it
    rf, X_test, y_test, y_pred, mse, mae, r2 = train_model(X, y)
    save_compressed_model(rf, 'model.pkl.gz')

    # Load the model from the compressed file
    rf = load_compressed_model('model.pkl.gz')

    # Sidebar for navigation
    page = st.sidebar.selectbox(
        'Navigate',
        ['Prediction', 'Model Performance', 'Dataset Insights']
    )
    
    if page == 'Prediction': 
        st.header('Course Rating Prediction')

        input_data = {} # dictionary to store user input data

        # Take user input
        available_titles = list(le_title.classes_)
        search_query = st.text_input('Search for a Course Title')
        filtered_titles = [title for title in available_titles if search_query.lower() in title.lower()] # filter titles
        selected_title = st.selectbox('Select Course Title', filtered_titles if search_query else available_titles)
        input_data['title'] = selected_title

        input_data['num_subscribers'] = st.number_input('Number of Subscribers', min_value=0, value=100)
        input_data['num_reviews'] = st.number_input('Number of Reviews', min_value=0, value=10)
        input_data['num_published_practice_tests'] = st.number_input('Number of Practice Tests', min_value=0, value=0)
        input_data['price_detail__amount'] = st.number_input('Course Price', min_value=0.0, value=10.0, step=0.1)

        wishlist_status = st.selectbox('Is Wishlisted', [False, True])
        input_data['is_wishlisted'] = wishlist_status

        input_df = pd.DataFrame([input_data])

        # Encoding
        input_df['title'] = le_title.transform([input_df['title'][0]])
        input_df['is_wishlisted'] = int(input_df['is_wishlisted'][0]) # convert boolean to integer [false: 0, true: 1]

        input_df = input_df[X.columns]
        
        # Scaling
        numerical_cols = ['num_subscribers', 'num_reviews', 'price_detail__amount']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Prediction
        if st.button('Predict Rating'):
            prediction = rf.predict(input_df)
            st.success(f'Predicted Course Rating: {prediction[0]:.2f}')

    elif page == 'Model Performance':
        st.header('Model Performance Metrics')
        col1, col2, col3 = st.columns(3)

        # Display model performance metrics
        with col1:
            st.metric('Mean Squared Error', f'{mse:.4f}')
        with col2:
            st.metric('Mean Absolute Error', f'{mae:.4f}')
        with col3:
            st.metric('R-squared', f'{r2:.4f}')

        # Display feature importance
        st.subheader('Feature Importance') 
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False) 

        st.bar_chart(feature_importance.set_index('feature'))

    elif page == 'Dataset Insights': 
        st.header('Dataset Overview')
        
        # Display the dataset summary statistics
        st.subheader('Dataset Statistics')
        st.write(X.describe())
        
        # Display the correlation matrix
        st.subheader('Feature Correlation')
        corr_matrix = X.corr()
        st.dataframe(corr_matrix)

# Run the app
if __name__ == '__main__':
    main()


# Run the app
if __name__ == '__main__':
    main()
