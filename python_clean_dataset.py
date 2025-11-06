import os
import cv2
from tqdm import tqdm

# --- Folder paths (your real locations) ---
smile_train_dir = r"C:\Users\DELL\OneDrive\Desktop\archive (3)\train"
smile_test_dir = r"C:\Users\DELL\OneDrive\Desktop\archive (3)\test"

age_train_dir = r"C:\Users\DELL\OneDrive\Desktop\agedataset\train"
age_test_dir = r"C:\Users\DELL\OneDrive\Desktop\agedataset\test"


# --- Output directories (inside your VS Code project) ---
smile_output_train = "Dataset/train_clean"
smile_output_test = "Dataset/test_clean"
age_output_train = "Age_dataset/train_clean"
age_output_test = "Age_dataset/test_clean"


def save_image(img_path, save_path, size=(100, 100)):
    """Reads, resizes, and saves image."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return
        img = cv2.resize(img, size)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
    except Exception as e:
        pass  # ignore corrupt files


def clean_and_resize(input_dir, output_dir, size=(100, 100)):
    """Clean and resize dataset images (handles both subfolders and direct images)."""
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    items = os.listdir(input_dir)
    for item in tqdm(items, desc=f"Cleaning {os.path.basename(input_dir)}"):
        item_path = os.path.join(input_dir, item)

        if os.path.isdir(item_path):
            # If the folder contains more folders or images
            for img_name in os.listdir(item_path):
                img_path = os.path.join(item_path, img_name)
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    save_path = os.path.join(output_dir, item, img_name)
                    save_image(img_path, save_path, size)
        elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
            # If images are directly inside input_dir
            save_path = os.path.join(output_dir, item)
            save_image(item_path, save_path, size)


# --- Clean Smile dataset ---
print("🧹 Cleaning Smile Dataset...")
clean_and_resize(smile_train_dir, smile_output_train)
clean_and_resize(smile_test_dir, smile_output_test)

# --- Clean Age dataset ---
print("\n🧹 Cleaning Age Dataset...")
clean_and_resize(age_train_dir, age_output_train)
clean_and_resize(age_test_dir, age_output_test)

print("\n✅ Dataset cleaning completed!")
