import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def create_model(data):
    # Drop the diagnosis column from the data and set it as the target variable
    x = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Scale the data from each column so each column has the same scale for easier processing
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Split the data to train one model and test the other
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
        )
    
    # Train the model by fitting x and y train data set onto logistic model graph
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Test the model with y_test (actual) against y_pred (ours) to see how well the model performs
    y_pred = model.predict(x_test)
    print("Accuracy of model: ", accuracy_score(y_test, y_pred))
    print("Classification_report: \n", classification_report(y_test, y_pred))

    return model, scaler



# Clean data by removing unnecessary columns and remapping diagnosis column
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

# Main function
def main():

    data = get_clean_data()

    model, scaler = create_model(data)

    # Save the model and scaler to be used in the API
    with open ('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
  




if __name__ == "__main__":
    main()