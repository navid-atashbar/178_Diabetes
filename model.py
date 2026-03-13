import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from process import clean_data

def evaluate_model(name, model, x_val, y_val):
    predictions = model.predict(x_val)
    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")
    print(classification_report(y_val, predictions))


def train_knn(x_tr, y_tr, x_val, y_val):
    #To do:
    #call clean data
    #predict the model
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(x_tr, y_tr)
    evaluate_model("K-Nearest Neighbors", knn, x_val, y_val)

def train_logistic(x_tr, y_tr, x_val, y_val):
    #To do:
    #call clean data
    #predict the model
    log_regression = LogisticRegression(max_iter=1000)
    log_regression.fit(x_tr, y_tr)
    evaluate_model("Logistic Regression", log_regression, x_val, y_val)

def train_nn(x_tr, y_tr, x_val, y_val):
    #To do:
    #call clean data
    #predict the model
    nn = MLPClassifier(hidden_layer_sizes=(64,32), max_iter = 500, random_state = 42)
    nn.fit(x_tr, y_tr)
    evaluate_model("Neural Network", nn, x_val, y_val)

if __name__ == '__main__':
    x_tr, x_val, x_te, y_tr, y_val, y_te = clean_data("./diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv")

    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_tr)
    x_val_scaled = scaler.transform(x_val)
    x_te_scaled = scaler.transform(x_te)

    train_knn(x_tr_scaled, y_tr, x_val_scaled, y_val)
    train_logistic(x_tr_scaled, y_tr, x_val_scaled, y_val)
    train_nn(x_tr_scaled, y_tr, x_val_scaled, y_val)