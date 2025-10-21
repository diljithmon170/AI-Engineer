from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data generator (assuming FER2013 folder structure)
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    'archive/train/',
    target_size=(48,48),
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    'archive/train/',
    target_size=(48,48),
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# Train model
model.fit(train_data, validation_data=val_data, epochs=25)
model.save('emotion_model.h5')
