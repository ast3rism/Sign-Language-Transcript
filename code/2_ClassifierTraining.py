# Training the Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os

# Load preprocessed data
data_dict = pickle.load(open(os.path.join('path/to/data/', 'data.pickle'), 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check dataset balance
unique_labels, counts = np.unique(labels, return_counts=True)
label_distribution = dict(zip(unique_labels, counts))
print("Label distribution : ", label_distribution)

# Split the data into training and test sets
print("\nSplitting Test and Train data ...")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, random_state=42)

# Initialize the Random Forest model
print("Initializing Random Forest Model ...")
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

# Train the model
print("Training the model ...")
model.fit(x_train, y_train)

# Calculate training accuracy to check for overfitting
train_accuracy = accuracy_score(y_train, model.predict(x_train))
print(f'Training Accuracy: {train_accuracy * 100:.2f}%')

# Make predictions
y_predict = model.predict(x_test)

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

cm = input("Generate Confusion Matrix?\n")
if cm.lower() in ['yes', 'y']:
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_predict, labels=np.unique(labels))
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
else:
    pass

acc = input("Generate Accuracy & Cross Validation score?\n")
if acc.lower() in ['yes', 'y']:
    # Calculate accuracy
    score = accuracy_score(y_predict, y_test)
    cv_scores = cross_val_score(model, data, labels, cv=5)
    print(f'Cross-validation scores: {cv_scores}')
    print(f'Mean cross-validation accuracy: {np.mean(cv_scores) * 100:.2f}%')
    print(f'{score * 100:.2f}% of samples were classified correctly.')
else:
    pass