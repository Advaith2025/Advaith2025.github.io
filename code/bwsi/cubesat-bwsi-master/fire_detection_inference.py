import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

# Import your model architecture
class SimplifiedFireDetectionCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(SimplifiedFireDetectionCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        
        return x.squeeze(1)

# Configuration class (simplified from your original)
class Config:
    # Image dimensions
    ORIGINAL_WIDTH = 4608
    ORIGINAL_HEIGHT = 2592
    
    # Tiling configuration
    TILE_ROWS = 2
    TILE_COLS = 2
    
    # Derived tile dimensions
    TILE_WIDTH = ORIGINAL_WIDTH // TILE_COLS
    TILE_HEIGHT = ORIGINAL_HEIGHT // TILE_ROWS
    
    # Downsampling factor
    DOWNSAMPLE_FACTOR = 2
    
    # Threshold for considering a tile as containing fire
    FIRE_PIXEL_THRESHOLD = 1
    
    # Path configuration
    IMAGES_DIR = "fire_training_data/images"
    MASKS_DIR = "fire_training_data/masks"
    MODEL_SAVE_PATH = "fire_detection_model.pth"
    
    # File patterns
    IMAGE_PATTERN = "*_true_combined.png"
    MASK_PATTERN = "*_mask_combined.png"
    
    # Preprocessing
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(model_path, device):
    """Load the trained model"""
    model = SimplifiedFireDetectionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def run_inference(model, image_path, config):
    """
    Run inference on a single satellite image and return fire detection results.
    Returns a heatmap showing where fires are detected and confidence scores.
    """
    # Load the image
    img = Image.open(image_path).convert("RGB")
    
    # Prepare transform for inference
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    # Create empty heatmaps for results
    heatmap_scores = np.zeros((config.TILE_ROWS, config.TILE_COLS))
    heatmap_binary = np.zeros((config.TILE_ROWS, config.TILE_COLS))
    
    # Process each tile
    with torch.no_grad():
        for row in range(config.TILE_ROWS):
            for col in range(config.TILE_COLS):
                # Calculate tile coordinates
                left = col * config.TILE_WIDTH
                upper = row * config.TILE_HEIGHT
                right = left + config.TILE_WIDTH
                lower = upper + config.TILE_HEIGHT
                
                # Extract tile
                img_tile = img.crop((left, upper, right, lower))
                
                # Downsample
                new_width = config.TILE_WIDTH // config.DOWNSAMPLE_FACTOR
                new_height = config.TILE_HEIGHT // config.DOWNSAMPLE_FACTOR
                img_tile = img_tile.resize((new_width, new_height), Image.BILINEAR)
                
                # Apply transform
                img_tile = transform(img_tile).unsqueeze(0).to(config.DEVICE)
                
                # Get prediction
                output = model(img_tile).item()
                
                # Convert logit to probability with sigmoid
                prob = 1 / (1 + np.exp(-output))
                
                heatmap_scores[row, col] = prob
                heatmap_binary[row, col] = 1 if output > 0 else 0
    
    return heatmap_scores, heatmap_binary

def get_ground_truth(mask_path, config):
    """
    Extract ground truth from mask
    """
    if not os.path.exists(mask_path):
        print(f"Warning: Mask not found: {mask_path}")
        return np.zeros((config.TILE_ROWS, config.TILE_COLS))
    
    mask = np.array(Image.open(mask_path).convert("L"))
    ground_truth = np.zeros((config.TILE_ROWS, config.TILE_COLS))
    
    for row in range(config.TILE_ROWS):
        for col in range(config.TILE_COLS):
            start_y = row * config.TILE_HEIGHT
            end_y = start_y + config.TILE_HEIGHT
            start_x = col * config.TILE_WIDTH
            end_x = start_x + config.TILE_WIDTH
            
            mask_tile = mask[start_y:end_y, start_x:end_x]
            ground_truth[row, col] = 1 if np.sum(mask_tile > 0) >= config.FIRE_PIXEL_THRESHOLD else 0
    
    return ground_truth

def visualize_results(image_path, mask_path, heatmap_scores, heatmap_binary, output_path, config):
    """
    Visualize fire detection results with ground truth and model prediction
    """
    # Load the original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get ground truth from mask
    ground_truth = get_ground_truth(mask_path, config)
    
    # Create a heatmap overlay for confidence scores
    heatmap_img = np.zeros((config.ORIGINAL_HEIGHT, config.ORIGINAL_WIDTH), dtype=np.float32)
    
    # Fill the heatmap with confidence scores
    for row in range(config.TILE_ROWS):
        for col in range(config.TILE_COLS):
            y_start = row * config.TILE_HEIGHT
            y_end = (row + 1) * config.TILE_HEIGHT
            x_start = col * config.TILE_WIDTH
            x_end = (col + 1) * config.TILE_WIDTH
            
            heatmap_img[y_start:y_end, x_start:x_end] = heatmap_scores[row, col]
    
    # Normalize and convert to color
    heatmap_img = (heatmap_img * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
    
    # Overlay on the original image
    alpha = 0.4
    overlay = cv2.addWeighted(img, 1-alpha, heatmap_color, alpha, 0)
    
    # Create a figure with subplots
    plt.figure(figsize=(20, 15))
    
    # Plot original image
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot prediction overlay
    plt.subplot(2, 2, 2)
    plt.imshow(overlay)
    plt.title("Fire Detection Results (Confidence)")
    
    # Add text indicating fire probability for each tile
    for row in range(config.TILE_ROWS):
        for col in range(config.TILE_COLS):
            y_center = int((row + 0.5) * config.TILE_HEIGHT)
            x_center = int((col + 0.5) * config.TILE_WIDTH)
            
            # Add prediction probability
            plt.text(x_center, y_center, 
                     f"{heatmap_scores[row, col]:.2f}", 
                     color='white', 
                     fontsize=12, 
                     fontweight='bold',
                     horizontalalignment='center',
                     verticalalignment='center',
                     bbox=dict(facecolor='black', alpha=0.5))
    
    plt.axis('off')
    
    # Plot ground truth
    plt.subplot(2, 2, 3)
    plt.imshow(img)
    plt.title("Ground Truth (Fire Areas)")
    
    # Add text showing ground truth
    for row in range(config.TILE_ROWS):
        for col in range(config.TILE_COLS):
            y_center = int((row + 0.5) * config.TILE_HEIGHT)
            x_center = int((col + 0.5) * config.TILE_WIDTH)
            
            # Draw rectangle for ground truth
            y_start = row * config.TILE_HEIGHT
            y_end = (row + 1) * config.TILE_HEIGHT
            x_start = col * config.TILE_WIDTH
            x_end = (col + 1) * config.TILE_WIDTH
            
            if ground_truth[row, col] > 0:
                plt.gca().add_patch(plt.Rectangle((x_start, y_start), 
                                                 config.TILE_WIDTH, 
                                                 config.TILE_HEIGHT, 
                                                 fill=False, 
                                                 edgecolor='red', 
                                                 linewidth=3))
                plt.text(x_center, y_center, 
                         "FIRE", 
                         color='red', 
                         fontsize=12, 
                         fontweight='bold',
                         horizontalalignment='center',
                         verticalalignment='center')
    
    plt.axis('off')
    
    # Plot comparison (prediction vs ground truth)
    plt.subplot(2, 2, 4)
    plt.imshow(img)
    plt.title("Comparison: Prediction vs Ground Truth")
    
    # Add visualization for comparison
    for row in range(config.TILE_ROWS):
        for col in range(config.TILE_COLS):
            y_center = int((row + 0.5) * config.TILE_HEIGHT)
            x_center = int((col + 0.5) * config.TILE_WIDTH)
            
            # Draw rectangle with different colors based on prediction status
            y_start = row * config.TILE_HEIGHT
            y_end = (row + 1) * config.TILE_HEIGHT
            x_start = col * config.TILE_WIDTH
            x_end = (col + 1) * config.TILE_WIDTH
            
            pred = heatmap_binary[row, col]
            gt = ground_truth[row, col]
            
            # True positive (green), True negative (no box), False positive (yellow), False negative (red)
            if pred == 1 and gt == 1:  # True positive
                plt.gca().add_patch(plt.Rectangle((x_start, y_start), 
                                                 config.TILE_WIDTH, 
                                                 config.TILE_HEIGHT, 
                                                 fill=False, 
                                                 edgecolor='green', 
                                                 linewidth=3))
                status_text = "TP"
                text_color = 'green'
            elif pred == 0 and gt == 0:  # True negative
                status_text = "TN"
                text_color = 'blue'
            elif pred == 1 and gt == 0:  # False positive
                plt.gca().add_patch(plt.Rectangle((x_start, y_start), 
                                                 config.TILE_WIDTH, 
                                                 config.TILE_HEIGHT, 
                                                 fill=False, 
                                                 edgecolor='yellow', 
                                                 linewidth=3))
                status_text = "FP"
                text_color = 'yellow'
            else:  # False negative
                plt.gca().add_patch(plt.Rectangle((x_start, y_start), 
                                                 config.TILE_WIDTH, 
                                                 config.TILE_HEIGHT, 
                                                 fill=False, 
                                                 edgecolor='red', 
                                                 linewidth=3))
                status_text = "FN"
                text_color = 'red'
                
            plt.text(x_center, y_center, 
                     status_text, 
                     color=text_color, 
                     fontsize=12, 
                     fontweight='bold',
                     horizontalalignment='center',
                     verticalalignment='center',
                     bbox=dict(facecolor='black', alpha=0.5))
    
    plt.axis('off')
    
    # Save or display
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        output_path = "fire_detection_results.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    plt.close()
    
    return output_path

def evaluate_model(model, config, num_samples=20):
    """
    Evaluate model on a set of images and return metrics
    """
    # Get a list of image files
    image_files = sorted(glob.glob(os.path.join(config.IMAGES_DIR, config.IMAGE_PATTERN)))
    
    # Sample a subset of images if needed
    if num_samples < len(image_files):
        # Use fixed seed for reproducibility
        np.random.seed(42)
        image_files = np.random.choice(image_files, num_samples, replace=False)
    
    # Metrics
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    # Process each image
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Evaluating model on {len(image_files)} images...")
    for i, img_path in enumerate(image_files):
        # Get corresponding mask path
        base_name = os.path.basename(img_path).replace("_true_combined.png", "")
        mask_path = os.path.join(config.MASKS_DIR, f"{base_name}_mask_combined.png")
        
        # Run inference
        try:
            scores, binary = run_inference(model, img_path, config)
            
            # Get ground truth
            ground_truth = get_ground_truth(mask_path, config)
            
            # Update metrics (tile-level)
            for row in range(config.TILE_ROWS):
                for col in range(config.TILE_COLS):
                    pred = binary[row, col]
                    gt = ground_truth[row, col]
                    
                    if pred == 1 and gt == 1:
                        true_positives += 1
                    elif pred == 0 and gt == 0:
                        true_negatives += 1
                    elif pred == 1 and gt == 0:
                        false_positives += 1
                    else:  # pred == 0 and gt == 1
                        false_negatives += 1
            
            # Visualize results for a subset of images
            if i < 15:  # Save visualizations for first 5 images
                output_path = os.path.join(output_dir, f"eval_{i}_{base_name}.png")
                visualize_results(img_path, mask_path, scores, binary, output_path, config)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Calculate metrics
    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print metrics
    print("\nEvaluation Results:")
    print(f"Total Tiles: {total}")
    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = np.array([
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ])
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    classes = ['No Fire', 'Fire']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save confusion matrix
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    print(f"Saved confusion matrix to {confusion_matrix_path}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

if __name__ == "__main__":
    # Initialize configuration
    cfg = Config()
    
    # Create output directory
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Load model
    print(f"Loading model from {cfg.MODEL_SAVE_PATH}")
    # model = load_model(cfg.MODEL_SAVE_PATH, cfg.DEVICE)
    model = load_model("fire_detection_model.pth", cfg.DEVICE)
    
    # Evaluate model performance
    metrics = evaluate_model(model, cfg, num_samples=20)
    
    # Save metrics to file
    with open("evaluation_results/metrics.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print("\nEvaluation complete. Results saved to evaluation_results/ directory.")
