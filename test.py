import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

schema_df = pd.read_csv('C:/Users/stupa/Downloads/hackathon-schema.csv')
meeting_rooms_df = pd.read_csv('C:/Users/stupa/Downloads/meeting-rooms.csv')

schema_df['date'] = pd.to_datetime(schema_df['date'], format='%d/%m/%Y')
schema_df['day_of_week'] = schema_df['date'].dt.dayofweek
meeting_rooms_df['date'] = pd.to_datetime(meeting_rooms_df['date'], format='%d/%m/%Y')
meeting_rooms_df['day_of_week'] = meeting_rooms_df['date'].dt.dayofweek

X_meeting_rooms = meeting_rooms_df.drop(['row', 'date'], axis=1)
y_meeting_rooms = meeting_rooms_df[['nineToEleven', 'elevenToOne', 'oneToThree', 'threeToFive']].any(axis=1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_meeting_rooms, y_meeting_rooms, test_size=0.2, random_state=42
)

preprocessor_meeting_rooms = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'), ['room'])
], remainder='passthrough')

X_train_transformed = preprocessor_meeting_rooms.fit_transform(X_train)
X_test_transformed = preprocessor_meeting_rooms.transform(X_test)

X_train_transformed = X_train_transformed.astype('float32')
X_test_transformed = X_test_transformed.astype('float32')

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_transformed.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_transformed, y_train, epochs=20, batch_size=42, verbose=1)

test_loss, test_accuracy = model.evaluate(X_test_transformed, y_test, verbose=0)
print(f'Test Accuracy: {test_accuracy}')

predicted_occupancy_probabilities = model.predict(X_test_transformed)
predicted_occupancy = (predicted_occupancy_probabilities > 0.5).astype(int).flatten()

print(f"Actual Occupancy: {y_test.values}")
print(f"Predicted Occupancy: {predicted_occupancy}")
accuracy = accuracy_score(y_test, predicted_occupancy)
print(f"Accuracy of Predicted Occupancy: {accuracy}")

model.save('E:/Joc/Hackaton_ai/trained_model.h5')

import joblib
joblib.dump(preprocessor_meeting_rooms, 'E:/Joc/Hackaton_ai/preprocessor.pkl')