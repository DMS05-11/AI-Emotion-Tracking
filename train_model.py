import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image size & batch size
IMG_SIZE = (48, 48)
BATCH_SIZE = 32

# Load dataset
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale", class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test', target_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale", class_mode="categorical"
)

# Build CNN Model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 Emotion Classes
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=test_generator, epochs=10)

# Save the trained model
model.save('emotion_model.h5')

print("Training Complete! Model saved as 'emotion_model.h5'")
