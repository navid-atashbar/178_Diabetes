import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
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
    best_k = 5
    accuracy_best  = 0
    for k in [3,5,7,11,15]:

        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(x_tr, y_tr)
        acc = accuracy_score(y_val, knn.predict(x_val))
        print(f"K-NN Accuracy K={k}: {acc:.4f}")
        if acc > accuracy_best:
            accuracy_best = acc
            best_k = k
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(x_tr, y_tr)
    evaluate_model("K-Nearest Neighbors", knn, x_val, y_val)

    filename = 'knn_model.pkl'
    joblib.dump(knn, filename)
    print(f"Model saved to {filename}")

def train_logistic(x_tr, y_tr, x_val, y_val):
    #To do:
    #call clean data
    #predict the model
    log_regression = LogisticRegression(max_iter=1000,class_weight='balanced',C=0.1)
    log_regression.fit(x_tr, y_tr)
    evaluate_model("Logistic Regression", log_regression, x_val, y_val)

    filename = 'logistic_reg_model.pkl'
    joblib.dump(log_regression, filename)
    print(f"Model saved to {filename}")

def train_nn(x_tr, y_tr, x_val, y_val):
    #To do:
    #call clean data
    #predict the model
    nn = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter = 500, early_stopping=True,validation_fraction=0.1,random_state = 42)
    nn.fit(x_tr, y_tr)
    evaluate_model("Neural Network", nn, x_val, y_val)

    filename = 'nn_model.pkl'
    joblib.dump(nn, filename)
    print(f"Model saved to {filename}")

def train_rf(x_tr, y_tr, x_val, y_val):
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,class_weight='balanced',random_state=42)
    rf.fit(x_tr, y_tr)
    evaluate_model("Random Forest", rf, x_val, y_val)

    filename = 'rf_model.pkl'
    joblib.dump(rf, filename)
    print(f"Model saved to {filename}")

if __name__ == '__main__':
    x_tr, x_val, x_te, y_tr, y_val, y_te = clean_data("./diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv")

    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_tr)
    x_val_scaled = scaler.transform(x_val)
    x_te_scaled = scaler.transform(x_te)

    train_knn(x_tr_scaled, y_tr, x_val_scaled, y_val)
    train_logistic(x_tr_scaled, y_tr, x_val_scaled, y_val)
    train_nn(x_tr_scaled, y_tr, x_val_scaled, y_val)
    train_rf(x_tr_scaled, y_tr, x_val_scaled, y_val)