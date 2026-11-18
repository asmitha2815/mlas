
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read dataset
data = pd.read_csv(r'C:\ml\dataset\student1.csv')

# Remove missing values
data = data.dropna()

# X = Input features
X = data[['study_hours', 'attendance', 'previous_marks']]

# Y = Output
y = data['result']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.80, random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print("Accuracy:", accuracy * 100, "%")

# Test with new student data
new_student = pd.DataFrame(
    [[100, 100, 100]],
    columns=['study_hours', 'attendance', 'previous_marks']
)

prediction = model.predict(new_student)

print("Student Result:", prediction[0])
