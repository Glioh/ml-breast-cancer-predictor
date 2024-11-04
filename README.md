# Breast Cancer Diagnosis Dashboard

An interactive dashboard for predicting breast cancer diagnosis using a machine learning model trained on patient cytology data. This application predicts the likelihood of a breast mass being benign or malignant based on tissue sample measurements.

## What This Does

This project leverages a machine learning pipeline to classify breast masses as either benign or malignant based on various features from cytology reports. Using logistic regression, the model achieves predictions that aid in early diagnosis.

### Data Preparation and Model Training

1. **Data Preprocessing**: Unnecessary columns are dropped, and diagnosis labels are mapped as follows: **Malignant (M)** = 1, **Benign (B)** = 0.
2. **Feature Scaling**: All input features are standardized to ensure uniform scale and improve model performance.
3. **Model Training**: Logistic regression is applied to classify masses, with the model trained on 80% of the data and tested on the remaining 20% for accuracy.

### Model Deployment and Prediction

Using **Streamlit**, this dashboard allows users to input sample features and obtain predictions in real-time. It displays the probabilities for benign and malignant diagnoses alongside a radar chart to visualize input feature distributions.

### Example Code

The model is trained as follows:

```python
# Load and preprocess data
data = get_clean_data()

# Train the logistic regression model
model, scaler = create_model(data)

# Save the model and scaler
with open ('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```
