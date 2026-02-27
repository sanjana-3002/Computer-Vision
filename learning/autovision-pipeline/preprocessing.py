"""
Interview Q&A — Day 2:

Q: What's the difference between Gaussian and Bilateral filter?
A: Gaussian blurs everything uniformly including edges.
   Bilateral blurs noise but detects edges and preserves them.
   For charts, bilateral is better because candle edges are important signal.

Q: When would you use Median blur over Gaussian?
A: When you have salt-and-pepper noise (random isolated black/white pixels).
   Median replaces each pixel with the neighborhood median,
   so outlier pixels get completely eliminated rather than just averaged down.

Q: Why do we filter BEFORE edge detection, not after?
A: Edge detectors (like Canny) work by finding pixel intensity gradients.
   Noise creates false gradients → false edges.
   Filtering first removes noise so only real structural edges are detected.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

class ChartPreprocessor:
    """
    Day 1 + Day 2: Image representations, color spaces, and filtering
    """

    def __init__(self, image_path: str):
        # OpenCV reads images in BGR by default (not RGB)
        # BGR = Blue, Green, Red — historical reason from early camera hardware
        self.bgr_image = cv2.imread(image_path)
        self.image_path = image_path

        if self.bgr_image is None:
            raise ValueError(f"Could not load image from {image_path}")

        print(f"Image loaded successfully")
        print(f"Shape: {self.bgr_image.shape}")   # (height, width, channels)
        print(f"dtype: {self.bgr_image.dtype}")   # uint8 = values 0-255
        print(f"Min pixel value: {self.bgr_image.min()}")
        print(f"Max pixel value: {self.bgr_image.max()}")

    # ------------------------------------------------------------------
    # DAY 1
    # ------------------------------------------------------------------

    def explore_as_array(self):
        """
        Most important concept in CV:
        An image is just a NumPy array of numbers.
        """
        img = self.bgr_image

        print("\n--- Image as NumPy Array ---")
        print(f"Height:   {img.shape[0]} pixels")
        print(f"Width:    {img.shape[1]} pixels")
        print(f"Channels: {img.shape[2]} (Blue, Green, Red)")

        print(f"\nTop-left 3x3 pixel values (BGR):")
        print(img[0:3, 0:3])

        pixel = img[100, 100]
        print(f"\nPixel at (100,100): Blue={pixel[0]}, Green={pixel[1]}, Red={pixel[2]}")

        return img

    def convert_color_spaces(self):
        """
        Different color spaces reveal different information.
        HSV  → isolate specific colors (red/green candles)
        Gray → remove color, keep structure
        LAB  → separate brightness from color (good for model training)
        """
        rgb  = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2HSV)
        lab  = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2LAB)

        print(f"\nGrayscale shape: {gray.shape}")  # 2D — no channel dimension
        return rgb, gray, hsv, lab

    def resize_and_crop(self, target_size=(224, 224)):
        """
        224x224 is the standard ResNet input size.
        INTER_AREA is best for shrinking (less aliasing).
        """
        resized = cv2.resize(
            self.bgr_image,
            target_size,            # cv2.resize takes (width, height) — note: flipped vs shape!
            interpolation=cv2.INTER_AREA
        )
        print(f"\nOriginal size: {self.bgr_image.shape[:2]}")
        print(f"Resized to:    {resized.shape[:2]}")

        h, w = self.bgr_image.shape[:2]
        center_crop = self.bgr_image[
            h//4 : 3*h//4,   # middle 50% vertically   (rows = height)
            w//4 : 3*w//4    # middle 50% horizontally  (cols = width)
        ]
        print(f"Center crop:   {center_crop.shape[:2]}")
        return resized, center_crop

    def visualize_all(self):
        """
        Show all Day 1 representations in one plot.
        Always visualize — this is how you catch bugs early.
        """
        rgb, gray, hsv, lab = self.convert_color_spaces()
        resized, crop = self.resize_and_crop()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Day 1: Image Representations', fontsize=16)

        axes[0, 0].imshow(rgb);                              axes[0, 0].set_title(f'Original RGB\n{rgb.shape}')
        axes[0, 1].imshow(gray, cmap='gray');                axes[0, 1].set_title(f'Grayscale\n{gray.shape}')
        axes[0, 2].imshow(hsv);                              axes[0, 2].set_title(f'HSV\n{hsv.shape}')
        axes[1, 0].imshow(lab);                              axes[1, 0].set_title(f'LAB\n{lab.shape}')
        axes[1, 1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)); axes[1, 1].set_title(f'Resized 224x224\n{resized.shape}')
        axes[1, 2].imshow(cv2.cvtColor(crop,    cv2.COLOR_BGR2RGB)); axes[1, 2].set_title(f'Center Crop\n{crop.shape}')

        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('day1_output.png', dpi=150, bbox_inches='tight')  # BUG FIX: save as .png not .webp
        plt.show()
        print("\nSaved: day1_output.png")

    # ------------------------------------------------------------------
    # DAY 2
    # ------------------------------------------------------------------

    def apply_filters(self):
        """
        Three filters, three different purposes:

        Gaussian Blur  → smooths everything uniformly (fast, simple)
        Median Blur    → kills salt-and-pepper noise, preserves edges better than Gaussian
        Bilateral      → best of both worlds: removes noise AND preserves edges

        For chart images: bilateral wins because candle edges = signal we care about.
        """
        # BUG FIX: was `cv2.cvtColor(image_path, ...)` — must use self.bgr_image
        gray = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2GRAY)

        # GAUSSIAN BLUR
        # Kernel (5,5) must be odd — needs a center pixel to anchor the kernel
        # sigmaX=0 → OpenCV auto-calculates sigma from kernel size
        gaussian = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

        # MEDIAN BLUR
        # Each pixel replaced by MEDIAN of its 5x5 neighborhood
        # Outlier pixels (noise) get voted out by their neighbors
        # Kernel is a single int, not a tuple
        median = cv2.medianBlur(gray, 5)

        # BILATERAL FILTER
        # d=9         → look at 9px diameter neighborhood
        # sigmaColor  → pixels must be within 75 intensity units to be blended
        # sigmaSpace  → pixels must be within 75 spatial units to be blended
        # Result: nearby pixels with similar color blend; edges (big color jump) are kept sharp
        bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        self._plot_filters(gray, gaussian, median, bilateral)
        return gaussian, median, bilateral

    def _plot_filters(self, original, gaussian, median, bilateral):
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
        plt.savefig('day2_filters.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: day2_filters.png")

    def compare_filter_edges(self):
        """
        Real test: which filter gives the cleanest edges on chart images?
        Run Canny AFTER each filter and compare visually.
        More clean, continuous edges = better filter for our pipeline.
        (Full Canny deep dive is Day 3 — today just observe the difference.)
        """
        gray = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2GRAY)

        gaussian  = cv2.GaussianBlur(gray, (5, 5), 0)
        median    = cv2.medianBlur(gray, 5)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

        edges_original  = cv2.Canny(gray,      50, 150)
        edges_gaussian  = cv2.Canny(gaussian,  50, 150)
        edges_median    = cv2.Canny(median,    50, 150)
        edges_bilateral = cv2.Canny(bilateral, 50, 150)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Day 2: Filter → Edge Detection Comparison', fontsize=16)

        for ax, img, title in zip(
            axes[0],
            [gray, gaussian, median, bilateral],
            ['Original', 'After Gaussian', 'After Median', 'After Bilateral']
        ):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        for ax, img, title in zip(
            axes[1],
            [edges_original, edges_gaussian, edges_median, edges_bilateral],
            ['Edges: Original', 'Edges: Gaussian', 'Edges: Median', 'Edges: Bilateral']
        ):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('day2_filter_edges.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: day2_filter_edges.png")


# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Pass your chart image path as a command line argument
    # Example: python opencv_pipeline.py data/raw/chart1.png
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/chart1.webp"

    processor = ChartPreprocessor(image_path)

    # Day 1
    processor.explore_as_array()
    processor.visualize_all()

    # Day 2
    processor.apply_filters()
    processor.compare_filter_edges()