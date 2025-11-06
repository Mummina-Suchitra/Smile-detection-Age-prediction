import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# --- Dataset Paths ---
train_dir = r"C:\Users\DELL\OneDrive\Desktop\Smile_detection_and_age_prediction2\Age_dataset\train_clean"
test_dir = r"C:\Users\DELL\OneDrive\Desktop\Smile_detection_and_age_prediction2\Age_dataset\test_clean"

# --- Verify Paths ---
if not os.path.exists(train_dir):
    print(f"❌ Train directory not found: {train_dir}")
if not os.path.exists(test_dir):
    print(f"❌ Test directory not found: {test_dir}")

print("✅ Paths verified. Starting Age Prediction Training...")

# --- Data Preprocessing ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

# --- Model Architecture ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')  # Multi-class
])

# --- Compile ---
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Train ---
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=test_gen
)

# --- Evaluate final accuracy ---
loss, acc = model.evaluate(test_gen)
print(f"\n✅ Final Test Accuracy: {acc:.2f}")
print(f"✅ Final Test Loss: {loss:.2f}")

# --- Save Model ---
os.makedirs("models", exist_ok=True)
model.save("models/age_model.h5")
print("\n✅ Age prediction model trained and saved successfully as 'models/age_model.h5'")
