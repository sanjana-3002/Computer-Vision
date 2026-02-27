import cv2
import numpy as np
import matplotlib.pyplot as plt

class ChartPreprocessor:
    """
    Day 1: Understanding images as arrays + color spaces
    This is the foundation of everything we build on top of
    """

    def __init__(self, image_path: str):
        # OpenCV reads images in BGR by default (not RGB)
        # BGR = Blue, Green, Red — historical reason from early camera hardware
        self.bgr_image = cv2.imread(image_path)
        self.image_path = image_path

        if self.bgr_image is None:
            raise ValueError(f"Could not load image from {image_path}")

        print(f"Image loaded successfully")
        print(f"Shape: {self.bgr_image.shape}")  # (height, width, channels)
        print(f"dtype: {self.bgr_image.dtype}")  # uint8 = values 0-255
        print(f"Min pixel value: {self.bgr_image.min()}")
        print(f"Max pixel value: {self.bgr_image.max()}")

    def explore_as_array(self):
        """
        This is the most important concept in CV:
        An image is just a NumPy array of numbers
        """
        img = self.bgr_image

        print("\n--- Image as NumPy Array ---")
        print(f"Height: {img.shape[0]} pixels")
        print(f"Width:  {img.shape[1]} pixels")
        print(f"Channels: {img.shape[2]} (Blue, Green, Red)")

        # Look at actual pixel values in top-left 3x3 region
        print(f"\nTop-left 3x3 pixel values (BGR):")
        print(img[0:3, 0:3])

        # A single pixel
        pixel = img[100, 100]
        print(f"\nPixel at (100,100): Blue={pixel[0]}, Green={pixel[1]}, Red={pixel[2]}")

        return img

    def convert_color_spaces(self):
        """
        Different color spaces reveal different information
        HSV is great for isolating colors (useful for colored chart lines)
        Grayscale removes color, keeps structure
        LAB separates brightness from color
        """
        # BGR to RGB (for matplotlib display)
        rgb = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2RGB)

        # BGR to Grayscale
        gray = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2GRAY)
        print(f"\nGrayscale shape: {gray.shape}")  # Notice: only 2D now, no channels

        # BGR to HSV (Hue, Saturation, Value)
        # Great for isolating specific colors in charts
        hsv = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2HSV)

        # BGR to LAB
        # L = lightness, A = green-red, B = blue-yellow
        lab = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2LAB)

        return rgb, gray, hsv, lab

    def resize_and_crop(self, target_size=(224, 224)):
        """
        224x224 is the standard input size for ResNet
        We'll need this later when we feed images to our model
        """
        # Resize — interpolation matters
        # INTER_AREA is best for shrinking (less aliasing)
        # INTER_LINEAR is best for enlarging
        resized = cv2.resize(
            self.bgr_image,
            target_size,
            interpolation=cv2.INTER_AREA
        )
        print(f"\nOriginal size: {self.bgr_image.shape[:2]}")
        print(f"Resized to: {resized.shape[:2]}")

        # Crop — just numpy array slicing
        # Format: image[y_start:y_end, x_start:x_end]
        h, w = self.bgr_image.shape[:2]
        center_crop = self.bgr_image[
            h//4 : 3*h//4,   # middle 50% vertically
            w//4 : 3*w//4    # middle 50% horizontally
        ]
        print(f"Center crop size: {center_crop.shape[:2]}")

        return resized, center_crop

    def visualize_all(self):
        """
        Show everything we've learned today in one plot
        This is how you debug CV pipelines visually
        """
        rgb, gray, hsv, lab = self.convert_color_spaces()
        resized, crop = self.resize_and_crop()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Day 1: Image Representations', fontsize=16)

        # Original
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title(f'Original RGB\n{rgb.shape}')
        axes[0, 0].axis('off')

        # Grayscale
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title(f'Grayscale\n{gray.shape}')
        axes[0, 1].axis('off')

        # HSV
        axes[0, 2].imshow(hsv)
        axes[0, 2].set_title(f'HSV\n{hsv.shape}')
        axes[0, 2].axis('off')

        # LAB
        axes[1, 0].imshow(lab)
        axes[1, 0].set_title(f'LAB\n{lab.shape}')
        axes[1, 0].axis('off')

        # Resized
        axes[1, 1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Resized 224x224\n{resized.shape}')
        axes[1, 1].axis('off')

        # Cropped
        axes[1, 2].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'Center Crop\n{crop.shape}')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig('data/raw/day1_output.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nVisualization saved to data/raw/day1_output.png")


# Run it
if __name__ == "__main__":
    import sys

    # Use a chart image you downloaded from TradingView
    # Or use any image to start
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/chart1.png"

    processor = ChartPreprocessor(image_path)
    processor.explore_as_array()
    processor.visualize_all()