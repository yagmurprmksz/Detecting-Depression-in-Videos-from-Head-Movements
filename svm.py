import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load data
y_matrix_path = '/Users/yagmu/PycharmProjects/capstone/y_matrix_modified.csv'
x_matrix_path = '/Users/yagmu/PycharmProjects/capstone/combined_histograms.csv'

y_matrix = pd.read_csv(y_matrix_path)
x_matrix = pd.read_csv(x_matrix_path)

# Align X and y by merging based on the index or a common column
combined_data = pd.merge(y_matrix, x_matrix, left_index=True, right_index=True)

# Separate features (X) and labels (y)
X = combined_data.iloc[:, 2:]  # Assuming features start from the 3rd column
y = combined_data['PHQ8_Binary']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM model
svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42, C=1, gamma='scale') #10^-3 - 10^3 arası dene
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)