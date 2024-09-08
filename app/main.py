import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# As the dataset is small, we can get away with importing data inside app
# In a real world scenario, you would want to export from model/main.py using pickle
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def add_sidebar():
    st.sidebar.title("Update Data")

    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    # Create an empty dictionary to store the input values
    input_dict = {}

    # Iterate through each label and add sliders for each feature in the dataset
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label, 
            min_value=float(0),
            # Taking the maximum value in the dataset for each slider
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict

# Takes in the input dictionary, looks for max/min possible values and
# Normalizes the input values and returns a dictionary with scaled values
def get_scaled_values(input_dict):
    data = get_clean_data()


    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_value = X[key].max()
        min_value = X[key].min()

        # Normalizing the value using the following formula...
        # Read more about normalization here: https://www.codecademy.com/article/normalization
        scaled_value = (value - min_value) / (max_value - min_value)

        scaled_dict[key] = scaled_value

    return scaled_dict
    
def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 
                  'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
      r=[
          input_data['radius_mean'],
          input_data['texture_mean'],
          input_data['perimeter_mean'],
          input_data['area_mean'],
          input_data['smoothness_mean'],
          input_data['compactness_mean'],
          input_data['concavity_mean'],
          input_data['concave points_mean'],
          input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']

      ],
      theta=categories,
      fill='toself',
      name='Patient Metrics' 
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=False,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    
    # Input data is a dictionary, we need to convert it to a numpy array
    input_array = np.array(list(input_data.values())).reshape(1, -1)
 
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)

     
    if prediction[0] == 0:
        outcome = "<span class='diagnosis benign'>Benign</span>"
    else:
        outcome = "<span class='diagnosis malignant'>Malignant</span>"


    st.header("Prediction")
    st.write(f"The cell cluster is: ", outcome, unsafe_allow_html=True)
    
    st.write("Benign Probability: ", "**{:.2f}%**".format(round(model.predict_proba(input_array_scaled)[0][0] * 100, 2)))
    st.write("Malignant Probability: ", "**{:.2f}%**".format(round(model.predict_proba(input_array_scaled)[0][1] * 100, 2)))
    
    st.write("This app can assist professionals in making a diagnosis but should not be used as a substitute for an actual diagnosis.")

def main():
    st.set_page_config(
        page_title="Breast Cancer Dashboard",
        page_icon=":hospital:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Diagnosis")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")

        col1, col2 = st.columns([4, 1])

        with col1:
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart, use_container_width=True, config={'staticPlot': True})

        with col2:
            with stylable_container(
                key="prediction-container",
                css_styles =["""
                {
                    background-color: #262730;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                    white-space: normal;
                    width: 100%;
                }


                """,
                """
                .diagnosis {
                    color: #fff;
                    padding: 0.2em 0.5em;
                    border-radius: 0.5em;
                }
                """,
                """
                .diagnosis.benign {
                    background-color: #4CAF50;
                }
                """,
                """
                .diagnosis.malignant {
                    background-color: #FF4B4B;
                }
                """]
            ):
                add_predictions(input_data)

            
                
            

if __name__ == '__main__':
    main()