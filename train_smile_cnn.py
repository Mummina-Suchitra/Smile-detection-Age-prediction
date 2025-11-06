import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Dataset Paths (update with your actual locations) ---
train_dir = r"C:\Users\DELL\OneDrive\Desktop\Smile_detection_and_age_prediction2\Dataset\train_clean"
test_dir = r"C:\Users\DELL\OneDrive\Desktop\Smile_detection_and_age_prediction2\Dataset\test_clean"

# --- Check that folders exist ---
if not os.path.exists(train_dir):
    print(f"❌ Train directory not found: {train_dir}")
if not os.path.exists(test_dir):
    print(f"❌ Test directory not found: {test_dir}")

print("✅ Paths verified. Starting training...")

# --- Image Data Generators (Normalization + Augmentation) ---
train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary'
)

# --- CNN Model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# --- Compile ---
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# --- Train the model ---
history = model.fit(train_gen, epochs=10, validation_data=test_gen)

# --- Evaluate final accuracy ---
loss, acc = model.evaluate(test_gen)
print(f"\n✅ Final Test Accuracy: {acc:.2f}")
print(f"✅ Final Test Loss: {loss:.2f}")

# --- Save the model ---
os.makedirs("models", exist_ok=True)
model.save("models/smile_model.h5")
print("✅ Smile detection model trained and saved successfully as 'models/smile_model.h5'")
