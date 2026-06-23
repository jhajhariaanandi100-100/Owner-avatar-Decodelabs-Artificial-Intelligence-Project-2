"""
Project 2: Predictive Phase - Data Classification Using AI
A simple supervised learning pipeline that loads a dataset, splits it,
trains a basic classifier, and evaluates model accuracy.
"""

import csv
import math
import os
import random
from collections import Counter

DATASET_PATH = "project2_dataset.csv"


def create_sample_dataset(path):
    """Create a small classification dataset as a CSV file."""
    header = ["sepal_length", "sepal_width", "petal_length", "label"]
    rows = [
        [5.1, 3.5, 1.4, "setosa"],
        [4.9, 3.0, 1.4, "setosa"],
        [5.0, 3.6, 1.4, "setosa"],
        [5.5, 2.4, 3.7, "versicolor"],
        [5.7, 2.8, 4.1, "versicolor"],
        [6.0, 2.9, 4.5, "versicolor"],
        [6.3, 3.3, 6.0, "virginica"],
        [6.5, 3.0, 5.8, "virginica"],
        [7.1, 3.0, 5.9, "virginica"],
        [5.4, 3.7, 1.5, "setosa"],
        [5.2, 3.4, 1.4, "setosa"],
        [5.8, 2.7, 4.1, "versicolor"],
        [6.1, 2.6, 5.6, "virginica"],
        [5.6, 3.0, 4.5, "versicolor"],
        [6.7, 3.1, 4.7, "versicolor"],
        [6.8, 3.2, 5.9, "virginica"],
        [5.0, 3.5, 1.3, "setosa"],
        [6.4, 2.8, 5.6, "virginica"],
    ]

    with open(path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Dataset created at {path} with {len(rows)} rows.")


def load_dataset(path):
    """Load the dataset from a CSV file into a structured list."""
    dataset = []

    with open(path, mode="r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            features = [float(row[feature]) for feature in ["sepal_length", "sepal_width", "petal_length"]]
            label = row["label"]
            dataset.append({"features": features, "label": label})

    return dataset


def describe_dataset(dataset):
    """Print a brief summary of the dataset and label distribution."""
    print("\nDATASET SUMMARY")
    print("-" * 40)
    print(f"Total examples: {len(dataset)}")
    print(f"Feature count : {len(dataset[0]['features'])}")

    label_counts = Counter(example["label"] for example in dataset)
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    print("\nSample rows:")
    for example in dataset[:5]:
        print(f"  features={example['features']} label={example['label']}")


def split_dataset(dataset, test_ratio=0.3, seed=42):
    """Shuffle and split the dataset into training and testing sets."""
    random.Random(seed).shuffle(dataset)
    split_index = int(len(dataset) * (1 - test_ratio))
    train_set = dataset[:split_index]
    test_set = dataset[split_index:]
    return train_set, test_set


def euclidean_distance(point_a, point_b):
    """Calculate Euclidean distance between two numeric feature vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))


def knn_predict(train_set, test_features, k=3):
    """Predict the label for a test example using the k-NN algorithm."""
    distances = []
    for example in train_set:
        distance = euclidean_distance(example["features"], test_features)
        distances.append((distance, example["label"]))

    distances.sort(key=lambda item: item[0])
    nearest_labels = [label for _, label in distances[:k]]
    predicted_label = Counter(nearest_labels).most_common(1)[0][0]
    return predicted_label


def evaluate_model(train_set, test_set, k=3):
    """Evaluate the classifier on the test set and return accuracy."""
    correct = 0
    predictions = []

    for example in test_set:
        predicted = knn_predict(train_set, example["features"], k=k)
        actual_label = example["label"]
        predictions.append({
            "features": example["features"],
            "actual": actual_label,
            "predicted": predicted,
        })
        if predicted == actual_label:
            correct += 1

    accuracy = correct / len(test_set) if test_set else 0.0
    return accuracy, predictions


def main():
    if not os.path.exists(DATASET_PATH):
        create_sample_dataset(DATASET_PATH)

    dataset = load_dataset(DATASET_PATH)
    describe_dataset(dataset)

    train_set, test_set = split_dataset(dataset, test_ratio=0.3, seed=2026)
    print("\nTRAIN/TEST SPLIT")
    print("-" * 40)
    print(f"Training examples: {len(train_set)}")
    print(f"Testing examples : {len(test_set)}")

    accuracy, predictions = evaluate_model(train_set, test_set, k=3)
    print("\nMODEL EVALUATION")
    print("-" * 40)
    print(f"Classifier: k-NN (k=3)")
    print(f"Accuracy : {accuracy:.2%}")

    print("\nTEST PREDICTIONS")
    print("-" * 40)
    for row in predictions:
        print(f"features={row['features']} actual={row['actual']} predicted={row['predicted']}")

    print("\nYou can extend this project by adding more examples, more features, or a different classifier.")


if __name__ == "__main__":
    main()
