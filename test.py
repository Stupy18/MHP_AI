import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Assuming you have already downloaded the CSV files to a known location
# Load the datasets
schema_df = pd.read_csv('C:/Users/stupa/Downloads/hackathon-schema.csv')
meeting_rooms_df = pd.read_csv('C:/Users/stupa/Downloads/meeting-rooms.csv')

# Convert date columns to datetime and extract day of the week for both datasets
schema_df['date'] = pd.to_datetime(schema_df['date'], format='%d/%m/%Y')
schema_df['day_of_week'] = schema_df['date'].dt.dayofweek
meeting_rooms_df['date'] = pd.to_datetime(meeting_rooms_df['date'], format='%d/%m/%Y')
meeting_rooms_df['day_of_week'] = meeting_rooms_df['date'].dt.dayofweek

# Prepare features and target for 'meeting_rooms_df'
X_meeting_rooms = meeting_rooms_df.drop(['row', 'date'], axis=1, errors='ignore')
y_meeting_rooms = meeting_rooms_df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']].any(axis=1).astype(int)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_meeting_rooms,
    y_meeting_rooms,
    test_size=0.2,
    random_state=42
)

# Define the preprocessing pipeline
preprocessor_meeting_rooms = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'), ['room'])
], remainder='passthrough')

# Fit the preprocessing pipeline
X_train_transformed = preprocessor_meeting_rooms.fit_transform(X_train)
X_test_transformed = preprocessor_meeting_rooms.transform(X_test)

X_train_transformed = X_train_transformed.astype('float32')
X_test_transformed = X_test_transformed.astype('float32')

# Build the Keras model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_transformed.shape[1],)),
    Dropout(0.2),  # Add dropout layer
    Dense(32, activation='relu'),
    Dropout(0.2),  # Add dropout layer
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model with dropout
model.fit(X_train_transformed, y_train, epochs=20, batch_size=42, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_transformed, y_test, verbose=0)
print(f'Test Accuracy: {test_accuracy}')

# Function to generate random data and time
def generate_random_datetime():
    random_date = datetime.now() + timedelta(days=random.randint(1, 30))
    random_time = random.randint(9, 16)
    random_datetime = random_date.replace(hour=random_time, minute=0, second=0)
    return random_datetime

# Generate random data and time
random_datetime = generate_random_datetime()

# Adjust `random_data` to include the attendance columns with dummy values
random_data = pd.DataFrame({
    'room': [random.choice(X_meeting_rooms['room'].unique())],
    'capacity': [random.randint(1, 100)],
    'day_of_week': [random_datetime.weekday()],
    'nineToEleven': [0], 'elevenToOne': [0], 'oneToThree': [0], 'threeToFive': [0],
    'attendanceNineToEleven': [0], 'attendanceElevenToOne': [0],
    'attendanceOneToThree': [0], 'attendanceThreeToFive': [0]
})

# Now `random_data_transformed` should transform without error
random_data_transformed = preprocessor_meeting_rooms.transform(random_data)
occupancy_probability = model.predict(random_data_transformed)

print("Randomly Generated Date and Time:", random_datetime)
print("Predicted Occupancy Probability:", occupancy_probability[0][0])
