import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = 'C:\\MLPROJECT\\project\\breast_cancer_analysis\\breast_cancer_analysis\\ML_model\\My_decision_tree_model.pkl'
model = joblib.load(model_path)

# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")

def main():
    # Set the title of the web app
    st.title('Breast Cancer Prediction')

    # Add a description
    st.write('Enter Information')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader('Patient Information')

        # Add input fields for features
        patient_name = st.text_input('Patient Name')
        year = st.selectbox('Year', [2019, 2020])
        tumor_size = st.slider('Tumor Size (cm)', 1, 14, 1)
        inv_nodes = st.selectbox('Inv-Nodes', [0, 1, 3])
        breast = st.selectbox('Breast', ['Left', 'Right'])
        metastasis = st.selectbox('Metastasis', ['No', 'Yes'])
        breast_quadrant = st.selectbox('Breast Quadrant', ['Lower inner', 'Lower outer', 'Upper inner', 'Upper outer'])
        history = st.selectbox('History', ['No', 'Yes'])
        diagnosis_result = st.selectbox('Diagnosis Result', ['Benign', 'Malignant'])
        age = st.slider('Age', 18, 100, 65)
        menopause = st.selectbox('Menopause', ['Pre', 'Post'])

    # Convert categorical inputs to numerical
    breast_col = {'Left': 'Breast_Left', 'Right': 'Breast_Right'}
    metastasis_col = {'No': 'Metastasis_0', 'Yes': 'Metastasis_1'}
    breast_quadrant_col = {'Lower inner': 'Breast Quadrant_Lower inner', 'Lower outer': 'Breast Quadrant_Lower outer',
                           'Upper inner': 'Breast Quadrant_Upper inner', 'Upper outer': 'Breast Quadrant_Upper outer'}
    history_col = {'No': 'History_0', 'Yes': 'History_1'}
    diagnosis_result_col = {'Benign': 'Diagnosis Result_Benign', 'Malignant': 'Diagnosis Result_Malignant'}
    menopause_col = {'Pre': 'Menopause_Pre', 'Post': 'Menopause_Post'}

    breast = breast_col.get(breast, 'Breast_Left')  # Default to Left breast if not found
    metastasis = metastasis_col.get(metastasis, 'Metastasis_0')  # Default to No metastasis if not found
    breast_quadrant = breast_quadrant_col.get(breast_quadrant, 'Breast Quadrant_Lower inner')  # Default to Lower inner if not found
    history = history_col.get(history, 'History_0')  # Default to No history if not found
    diagnosis_result = diagnosis_result_col.get(diagnosis_result, 'Diagnosis Result_Benign')  # Default to Benign if not found
    menopause = menopause_col.get(menopause, 'Menopause_Pre')  # Default to Pre menopause if not found

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Year_2019': [1 if year == 2019 else 0],
        'Year_2020': [1 if year == 2020 else 0],
        'Tumor Size (cm)_{}'.format(tumor_size): [1],
        'Inv-Nodes_{}'.format(inv_nodes): [1],
        breast: [1],
        metastasis: [1],
        breast_quadrant: [1],
        history: [1],
        diagnosis_result: [1],
        'Age': [age],
        menopause: [1]
    })

    # Ensure columns are in the same order as during model training
    input_data = input_data[expected_columns]

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            st.write(f'Prediction for {patient_name}: {"Breast Cancer" if prediction[0] == 1 else "No Breast Cancer"}')
            st.write(f'Probability of Breast Cancer: {probability:.2f}')

            # Plotting
            fig, axes = plt.subplots(3, 1, figsize=(8, 16))

            # Plot Breast Cancer probability
            sns.barplot(x=['No Breast Cancer', 'Breast Cancer'], y=[1 - probability, probability], ax=axes[0], palette=['green', 'red'])
            axes[0].set_title('Breast Cancer Probability')
            axes[0].set_ylabel('Probability')

            # Plot Age distribution
            sns.histplot(input_data['Age'], kde=True, ax=axes[1])
            axes[1].set_title('Age Distribution')

            # Plot Breast Cancer pie chart
            axes[2].pie([1 - probability, probability], labels=['No Breast Cancer', 'Breast Cancer'], autopct='%1.1f%%', colors=['green', 'red'])
            axes[2].set_title('Breast Cancer Pie Chart')

            # Display the plots
            st.pyplot(fig)

            # Provide recommendations
            if prediction[0] == 1:
                st.error(f"{patient_name} is likely to have Breast Cancer. Consider seeking medical attention and following a healthy lifestyle.")
            else:
                st.success(f"{patient_name} is likely to not have Breast Cancer. Keep up the good habits and maintain a healthy lifestyle.")

if __name__ == '__main__':
    main()
