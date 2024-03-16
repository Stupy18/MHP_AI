# Additional libraries for handling 'meeting_rooms_df'
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

import numpy as np

from TimeSlotAggregator import TimeSlotAggregator

# Load the datasets
schema_df = pd.read_csv('C:/Users/stupa/Downloads/hackathon-schema.csv')
meeting_rooms_df = pd.read_csv('C:/Users/stupa/Downloads/meeting-rooms.csv')

# Convert date columns to datetime and extract day of the week for both datasets
schema_df['date'] = pd.to_datetime(schema_df['date'], format='%d/%m/%Y')
schema_df['day_of_week'] = schema_df['date'].dt.dayofweek
meeting_rooms_df['date'] = pd.to_datetime(meeting_rooms_df['date'], format='%d/%m/%Y')
meeting_rooms_df['day_of_week'] = meeting_rooms_df['date'].dt.dayofweek

# Assuming specific hour columns are named as 'nineToEleven', 'elevenToOne', etc., in 'meeting_rooms_df'
time_columns = [col for col in meeting_rooms_df.columns if 'attendance' in col]

# Adjusting the preprocessing pipeline for 'meeting_rooms_df'
preprocessor_meeting_rooms = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'), ['room'])
], remainder='passthrough')

# Switch the main pipeline to use Logistic Regression
pipeline_meeting_rooms = Pipeline([
    ('preprocessor', preprocessor_meeting_rooms),
    ('timeslot_aggregator', TimeSlotAggregator(time_columns_indices=[0, 1, 2, 3])),
    ('model', LogisticRegression(random_state=42))
])

# Prepare features and target for 'meeting_rooms_df'
X_meeting_rooms = meeting_rooms_df.drop(['row', 'date'] + time_columns, axis=1, errors='ignore')
y_meeting_rooms = (meeting_rooms_df[time_columns].sum(axis=1) > 0).astype(int)

# Cleaning NaNs
mask = ~y_meeting_rooms.isna()
X_meeting_rooms_clean = X_meeting_rooms[mask]
y_meeting_rooms_clean = y_meeting_rooms[mask]

# Splitting and training
X_train_mr, X_test_mr, y_train_mr, y_test_mr = train_test_split(
    X_meeting_rooms_clean,
    y_meeting_rooms_clean,
    test_size=0.2,
    random_state=42
)

# Fit the pipeline with Logistic Regression
pipeline_meeting_rooms.fit(X_train_mr, y_train_mr)

# Predicting and evaluating with Logistic Regression
y_pred_mr = pipeline_meeting_rooms.predict(X_test_mr)
accuracy_mr = accuracy_score(y_test_mr, y_pred_mr)
f1_mr = f1_score(y_test_mr, y_pred_mr, average='weighted')

print(f"Accuracy for meeting_rooms_df with Logistic Regression: {accuracy_mr}, F1 Score: {f1_mr}")

train_sizes, train_scores, test_scores = learning_curve(
    pipeline_meeting_rooms, X_meeting_rooms_clean, y_meeting_rooms_clean, cv=5,
    scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# # Plot learning curves
# plt.plot(train_sizes, train_mean, label="Training score", color="r")
# plt.plot(train_sizes, test_mean, label="Cross-validation score", color="g")
#
# # Plot the std deviation as a transparent range at each training set size
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.1)
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="g", alpha=0.1)
#
# # Draw plot
# plt.title("Learning Curve")
# plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
# plt.tight_layout()
# plt.show()

# Create a StratifiedKFold object
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Apply cross-validation using the stratified folds
cv_scores = cross_val_score(pipeline_meeting_rooms, X_meeting_rooms_clean, y_meeting_rooms_clean, cv=stratified_cv, scoring='accuracy')

print("Stratified Cross-validation accuracy scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# Predicting on the training set
y_train_pred = pipeline_meeting_rooms.predict(X_train_mr)
train_accuracy = accuracy_score(y_train_mr, y_train_pred)
train_f1 = f1_score(y_train_mr, y_train_pred, average='weighted')

print(f"Training Accuracy: {train_accuracy}, Training F1 Score: {train_f1}")



# Function to generate random data and time
def generate_random_datetime():
    # Generate random date within a reasonable range
    random_date = datetime.now() + timedelta(days=random.randint(1, 30))
    # Generate random time within working hours (9 AM to 5 PM)
    random_time = random.randint(9, 16)
    # Combine date and time
    random_datetime = random_date.replace(hour=random_time, minute=0, second=0)
    return random_datetime

# Generate random data and time
random_datetime = generate_random_datetime()

# Prepare the data in the required format
random_data = pd.DataFrame({'date': [random_datetime.date()],
                            'day_of_week': [random_datetime.weekday()],
                            'nineToEleven': [random.randint(0, 1)],
                            'elevenToOne': [random.randint(0, 1)],
                            'oneToThree': [random.randint(0, 1)],
                            'threeToFive': [random.randint(0, 1)],
                            'room': [random.choice(meeting_rooms_df['room'])]})


random_data['capacity'] = random.randint(1, 100)  # Assuming capacity can vary from 1 to 100

# Predict occupancy percentage
occupancy_prediction = pipeline_meeting_rooms.predict(random_data)

print("Randomly Generated Date and Time:", random_datetime)
print("Predicted Occupancy Percentage:", occupancy_prediction[0] * 100, "%")




# Additional steps su ch as sanity check, cross-validation, and examining model predictions remain unchanged
