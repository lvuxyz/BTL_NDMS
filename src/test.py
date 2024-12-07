import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd

from src.color_recognition_api import color_histogram_feature_extraction, knn_classifier


def load_training_data(filename='training.data'):
    # Read the training data
    data = pd.read_csv(filename, header=None)

    # Separate features and labels
    X = data.iloc[:, :-1]  # All columns except the last
    y = data.iloc[:, -1]  # Last column is the label

    return X, y


def evaluate_knn_accuracy(X, y, test_size=0.2, random_state=42, k=7):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Prepare training and test data as lists for existing KNN implementation
    training_feature_vector = X_train.values.tolist()
    test_feature_vector = X_test.values.tolist()

    # Add labels to training feature vector for existing implementation
    for i in range(len(training_feature_vector)):
        training_feature_vector[i].append(y_train.iloc[i])

    # Predictions
    predictions = []
    for test_instance in test_feature_vector:
        neighbors = knn_classifier.kNearestNeighbors(training_feature_vector, test_instance, k)
        result = knn_classifier.responseOfNeighbors(neighbors)
        predictions.append(result)

    # Calculate metrics
    print("Detailed Classification Report:")
    print(classification_report(y_test, predictions))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Macro F1-score
    macro_f1 = f1_score(y_test, predictions, average='macro')
    print(f"\nMacro F1-Score: {macro_f1:.4f}")

    # Weighted F1-score
    weighted_f1 = f1_score(y_test, predictions, average='weighted')
    print(f"Weighted F1-Score: {weighted_f1:.4f}")

    return predictions


def main():
    # Ensure training data is created first
    color_histogram_feature_extraction.training()

    # Load data
    X, y = load_training_data()

    # Evaluate accuracy
    evaluate_knn_accuracy(X, y)


if __name__ == '__main__':
    main()