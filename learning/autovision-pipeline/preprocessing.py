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
        self.bgr_image = cv2.imread('/Users/sanjanawaghray/Documents/projects/Computer-Vision-1/learning/raw_image.webp')
        self.image_path = '/Users/sanjanawaghray/Documents/projects/Computer-Vision-1/learning/raw_image.webp'

        if self.bgr_image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
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
        plt.savefig('/Users/sanjanawaghray/Documents/projects/Computer-Vision-1/learning/raw_iamge.webp', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nVisualization saved to /Users/sanjanawaghray/Documents/projects/Computer-Vision-1/learning/raw_image.webp")

    def apply_filters(self):

        """
        Day 2: Three filters, three different purposes
        
        Gaussian Blur  → removes random electronic noise, smooths everything
        Median Blur    → removes salt-and-pepper noise, preserves edges better
        Bilateral      → removes noise BUT preserves edges (best for charts)
        
        Key insight: for chart analysis we care about edges (candle boundaries)
        so bilateral filter is our weapon of choice
        """
        
        gray = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2GRAY)
        
        # --- GAUSSIAN BLUR ---
        # Kernel size must be odd (3,5,7,9...) — why? Because it needs a center pixel
        # Higher kernel = more blur
        # sigmaX=0 means OpenCV calculates sigma automatically from kernel size
        gaussian = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)
        
        # --- MEDIAN BLUR ---
        # Replaces each pixel with the MEDIAN of its neighborhood
        # Great for salt-and-pepper noise (random black/white pixels)
        # Kernel must be odd integer, not a tuple
        median = cv2.medianBlur(gray, 5)
        
        # --- BILATERAL FILTER ---
        # The smart filter — blurs noise but PRESERVES edges
        # d=9: diameter of pixel neighborhood
        # sigmaColor=75: how much color difference is considered "same region"
        # sigmaSpace=75: how much spatial distance affects blending
        bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Visualize all three side by side
        self._plot_filters(gray, gaussian, median, bilateral)
        
        return gaussian, median, bilateral

    def _plot_filters(self, original, gaussian, median, bilateral):
        """
        Always visualize your preprocessing steps
        This is how you catch bugs early
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle('Day 2: Filtering Comparison', fontsize=16)
        
        images = [original, gaussian, median, bilateral]
        titles = [
            'Original Grayscale',
            'Gaussian Blur\n(smooths everything)',
            'Median Blur\n(removes speckles)',
            'Bilateral Filter\n(preserves edges)'
        ]
        
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('data/raw/day2_filters.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved to data/raw/day2_filters.png")

    def compare_filter_edges(self):
        """
        The real test: which filter preserves chart edges best?
        We check by running Canny edge detection AFTER each filter
        More clean edges = better filter for our use case
        
        (We'll learn Canny properly on Day 3 — today just see the difference)
        """
        gray = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2GRAY)
        
        gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
        median = cv2.medianBlur(gray, 5)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Quick Canny preview — don't worry about parameters yet, Day 3 covers this
        edges_original  = cv2.Canny(gray, 50, 150)
        edges_gaussian  = cv2.Canny(gaussian, 50, 150)
        edges_median    = cv2.Canny(median, 50, 150)
        edges_bilateral = cv2.Canny(bilateral, 50, 150)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Day 2: Filter → Edge Detection Comparison', fontsize=16)
        
        # Top row: filtered images
        for ax, img, title in zip(
            axes[0],
            [gray, gaussian, median, bilateral],
            ['Original', 'After Gaussian', 'After Median', 'After Bilateral']
        ):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        
        # Bottom row: edges after each filter
        for ax, img, title in zip(
            axes[1],
            [edges_original, edges_gaussian, edges_median, edges_bilateral],
            ['Edges: Original', 'Edges: Gaussian', 'Edges: Median', 'Edges: Bilateral']
        ):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('data/raw/day2_filter_edges.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved to data/raw/day2_filter_edges.png")

# Run it
if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/chart1.png"
    
    processor = ChartPreprocessor(image_path)
    
    # Day 1
    processor.explore_as_array()
    processor.visualize_all()
    
    # Day 2
    processor.apply_filters()
    processor.compare_filter_edges()