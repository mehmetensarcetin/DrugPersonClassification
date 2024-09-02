import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class DataPreprocessor:
    def __init__(self, dataframe):
        self.df = dataframe
        self.label_encoders = {}

    def encode_features(self):
        categorical_columns = ['Sex', 'BP', 'Cholesterol', 'Drug']
        for col in categorical_columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        return self.df

    def split_data(self, target, test_size=0.2, random_state=42):
        X = self.df.drop(target, axis=1)
        y = self.df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

class DrugPredictionModel:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        return accuracy, report

def main():
    # Verinin yüklenmesi
    df = pd.read_csv('drug200.csv')

    # Veri ön işleme
    preprocessor = DataPreprocessor(df)
    preprocessor.encode_features()
    X_train, X_test, y_train, y_test = preprocessor.split_data(target='Drug')

    # Modellerin tanımlanması
    models = [
        DrugPredictionModel(DecisionTreeClassifier(random_state=42), "Decision Tree"),
        DrugPredictionModel(RandomForestClassifier(random_state=42), "Random Forest"),
        DrugPredictionModel(LogisticRegression(max_iter=1700, random_state=42), "Logistic Regression")
    ]

    # Modellerin eğitilmesi ve değerlendirilmesi
    for model in models:
        model.train(X_train, y_train)
        accuracy, report = model.evaluate(X_test, y_test)
        print(f"Model: {model.model_name}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:\n", report)
        print("-" * 60)

if __name__ == "__main__":
    main()
