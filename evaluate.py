import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import insightface

# ==============================================================================
# FACIAL RECOGNITION EVALUATION SCRIPT
# ==============================================================================
# This script evaluates the accuracy of the InsightFace (buffalo_l) engine
# using your own dataset of faces.
#
# INSTRUCTIONS:
# 1. Create a folder named 'dataset' in this directory.
# 2. Inside 'dataset', create a folder for each person (e.g., 'person1', 'person2')
# 3. Put 5-10 images of that person in their folder.
# 4. Run this script!
# ==============================================================================

DATASET_DIR = "dataset"

def evaluate_system():
    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        print(f"Error: No dataset found at '{DATASET_DIR}'.")
        print("Please create a 'dataset' folder with subfolders for each person to evaluate.")
        # GENERATE DUMMY DATA FOR DEMONSTRATION IF FOLDER DOESNT EXIST
        generate_dummy_data()
    
    print("Loading InsightFace buffalo_l engine...")
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # 1. ENROLL THE FIRST IMAGE OF EACH PERSON (Gallery)
    gallery = {}
    true_labels = []
    pred_labels = []
    
    people = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    print("\n--- Enrolling Gallery ---")
    for person in people:
        person_dir = os.path.join(DATASET_DIR, person)
        images = os.listdir(person_dir)
        if not images:
            continue
            
        # Use first image as the enrolled identity
        enroll_img_path = os.path.join(person_dir, images[0])
        img = cv2.imread(enroll_img_path)
        if img is None: continue
        
        faces = app.get(img)
        if faces:
            gallery[person] = faces[0].embedding
            print(f"Enrolled: {person}")
    
    # 2. TEST THE REMAINING IMAGES (Probe)
    print("\n--- Running Evaluation ---")
    for person in people:
        person_dir = os.path.join(DATASET_DIR, person)
        images = os.listdir(person_dir)
        
        for img_name in images[1:]:  # Skip the first image (used for enrollment)
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            faces = app.get(img)
            if not faces:
                continue
            
            embedding = faces[0].embedding
            
            # Find best match in gallery
            best_match = "Unknown"
            best_score = -1.0
            
            for g_person, g_emb in gallery.items():
                # Cosine similarity
                sim = np.dot(embedding, g_emb) / (np.linalg.norm(embedding) * np.linalg.norm(g_emb))
                if sim > best_score:
                    best_score = sim
                    best_match = g_person
                    
            if best_score > 0.5: # Match threshold
                true_labels.append(person)
                pred_labels.append(best_match)
            else:
                true_labels.append(person)
                pred_labels.append("Unknown")
                
    # 3. CALCULATE METRICS
    if len(true_labels) == 0:
        print("Not enough testing images found to evaluate.")
        return
        
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\n======================================")
    print(f" SYSTEM ACCURACY: {accuracy * 100:.2f}%")
    print(f"======================================")
    
    # 4. PLOT CONFUSION MATRIX
    labels = list(gallery.keys()) + ["Unknown"]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Facial Recognition Confusion Matrix\nAccuracy: {accuracy*100:.2f}%')
    plt.ylabel('Actual Person')
    plt.xlabel('Predicted Person')
    
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    print(f"\nConfusion Matrix saved to: {plot_path}")
    plt.close()

def generate_dummy_data():
    """Generates synthetic noise matrices if no real images are provided, just to show how it works."""
    print("Generating synthetic dummy test data so the script doesn't crash...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    for i in range(1, 4):
        p_dir = os.path.join(DATASET_DIR, f"Person_{i}")
        os.makedirs(p_dir, exist_ok=True)
        # Create 5 dummy random images
        for j in range(5):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(p_dir, f"{j}.jpg"), img)
            
if __name__ == "__main__":
    evaluate_system()
