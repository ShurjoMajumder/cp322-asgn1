import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def load_data(real_path, fake_path):

    if not os.path.exists(real_path):
        raise FileNotFoundError(f"File not found: {real_path}")
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"File not found: {fake_path}")

  
    with open(real_path, "r", encoding="utf-8") as f:
        real_headlines = f.readlines()
    with open(fake_path, "r", encoding="utf-8") as f:
        fake_headlines = f.readlines()

    
    headlines = real_headlines + fake_headlines
    labels = [1] * len(real_headlines) + [0] * len(fake_headlines)

    
    headlines = [" ".join(line.split()) for line in headlines]

    
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(headlines)

    
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer


def select_model(X_train, y_train, X_val, y_val):
    max_depths = [2, 5, 10, 20, 50]  # Try different max_depth values
    val_accuracies = []

    for depth in max_depths:
        model = DecisionTreeClassifier(max_depth=depth, criterion="entropy", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        val_accuracies.append(acc)
        print(f"max_depth={depth}, Validation Accuracy={acc:.4f}")

   
    plt.figure(figsize=(8, 5))
    plt.plot(max_depths, val_accuracies, marker='o', linestyle='-', color='b', label="Validation Accuracy")
    plt.xlabel("max_depth")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. max_depth")
    plt.legend()
    plt.show()

    # Return the best max_depth
    best_depth = max_depths[np.argmax(val_accuracies)]
    return best_depth


def train_and_test(X_train, y_train, X_test, y_test, best_depth):
    model = DecisionTreeClassifier(max_depth=best_depth, criterion="entropy", random_state=42)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy (best_depth={best_depth}): {test_acc:.4f}")
    return model


def visualize_tree(model, vectorizer):
    plt.figure(figsize=(12, 6))
    plot_tree(model, feature_names=vectorizer.get_feature_names_out(), class_names=["Fake", "Real"], max_depth=2, filled=True)
    plt.title("Decision Tree Visualization (First Two Layers)")
    plt.show()


if __name__ == "__main__":
    real_path = "/workspaces/cp322/Decision Tree-Dataset/real.txt"
    fake_path = "/workspaces/cp322/Decision Tree-Dataset/fake.txt"


   
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = load_data(real_path, fake_path)

    
    best_depth = select_model(X_train, y_train, X_val, y_val)

   
    model = train_and_test(X_train, y_train, X_test, y_test, best_depth)

    
    visualize_tree(model, vectorizer)

def print_tree_rules(model, vectorizer):
    """
    Prints the first two layers of the decision tree in text format.
    """
    feature_names = vectorizer.get_feature_names_out()
    tree = model.tree_
    
    
    root_feature = feature_names[tree.feature[0]]
    print(f"Root Node: '{root_feature}' (Threshold: {tree.threshold[0]:.2f})")
    
    left_child = tree.children_left[0]
    right_child = tree.children_right[0]
    
    if left_child != -1:
        left_feature = feature_names[tree.feature[left_child]]
        print(f"Left Child: '{left_feature}' (Threshold: {tree.threshold[left_child]:.2f})")
    
    if right_child != -1:
        right_feature = feature_names[tree.feature[right_child]]
        print(f"Right Child: '{right_feature}' (Threshold: {tree.threshold[right_child]:.2f})")

print_tree_rules(model, vectorizer)


