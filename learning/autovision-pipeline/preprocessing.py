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

Interview Q&A — Day 4:

Q: What's the difference between erosion and dilation?
A: Erosion shrinks white regions — any white pixel touching a black pixel becomes black.
   Dilation expands white regions — any black pixel touching a white pixel becomes white.
   Think of erosion as "eating away" the edges of white blobs from outside in.

Q: When would you use opening vs closing?
A: Opening (erode then dilate) removes small isolated white dots (noise) without
   affecting larger continuous structures. Use it when Canny gives you speckle noise.
   Closing (dilate then erode) fills small gaps in white lines without changing their
   overall shape. Use it when Canny breaks up continuous edges — which happens a lot
   on stock chart candle wicks.

Q: Why use RETR_EXTERNAL instead of RETR_TREE for chart analysis?
A: RETR_TREE gives you every contour including nested ones (holes inside shapes).
   For individual candle detection we only care about the outer boundary of each region.
   RETR_EXTERNAL skips the inner contours and runs faster — less noise to filter out.

Q: What does CHAIN_APPROX_SIMPLE do and why does it matter?
A: It compresses contour representations by only storing the endpoints of straight segments.
   A rectangle stored with CHAIN_APPROX_NONE would have hundreds of points along each side.
   CHAIN_APPROX_SIMPLE stores just the 4 corners. Faster, less memory, same shape information.
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
    # DAY 3
    # ------------------------------------------------------------------


    def canny_deep_dive(self):

        gray = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2GRAY)

        # STEP 1: Gaussian blur — we do this manually so we can visualize it
        # Canny does this internally too, but doing it beforehand with
        # bilateral gives us cleaner results (as we proved on Day 2)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # STEP 2: Sobel gradients — manually compute what Canny does internally
        # cv2.CV_64F means output is float64 — important because gradients
        # can be negative (dark-to-light vs light-to-dark)
        # if you use uint8 (0-255), negative gradients get clipped to 0 and you miss edges
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # vertical edges
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # horizontal edges

        # Convert to absolute values — we care about magnitude, not direction sign
        sobel_x_abs = cv2.convertScaleAbs(sobel_x)
        sobel_y_abs = cv2.convertScaleAbs(sobel_y)

        # Combine X and Y into gradient magnitude
        gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

        # Normalize to 0-255 for visualization
        gradient_magnitude = cv2.normalize(
            gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # STEP 3 + 4: Full Canny (NMS + Hysteresis happen inside)
        canny_tight  = cv2.Canny(blurred, 100, 300)  # high thresholds — fewer, stronger edges
        canny_medium = cv2.Canny(blurred,  50, 150)  # balanced — our default
        canny_loose  = cv2.Canny(blurred,  20,  60)  # low thresholds — more edges, more noise

        self._plot_canny_steps(
            gray, blurred, sobel_x_abs, sobel_y_abs,
            gradient_magnitude, canny_tight, canny_medium, canny_loose
        )

        return canny_medium  # return the balanced one for use in later pipeline

    def _plot_canny_steps(self, gray, blurred, sobel_x, sobel_y,
                          gradient, tight, medium, loose):
        """
        Visualize every step of the Canny algorithm.
        Row 1: intermediate steps inside Canny
        Row 2: effect of different threshold values
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Day 3: Canny Deep Dive', fontsize=16)

        step_images = [gray, blurred, sobel_x, gradient]
        step_titles = [
            'Original Grayscale',
            'Step 1: Gaussian Blur\n(noise removed)',
            'Step 2: Sobel X\n(vertical edges)',
            'Step 2: Gradient Magnitude\n(Gx² + Gy² combined)'
        ]

        for ax, img, title in zip(axes[0], step_images, step_titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        thresh_images = [sobel_y, tight, medium, loose]
        thresh_titles = [
            'Step 2: Sobel Y\n(horizontal edges)',
            'Canny tight (100/300)\nfewer, stronger edges',
            'Canny medium (50/150)\nbalanced ← our choice',
            'Canny loose (20/60)\nmore edges + noise'
        ]

        for ax, img, title in zip(axes[1], thresh_images, thresh_titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('day3_canny.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: day3_canny.png")

    def thresholding(self):
        """
        Day 3: Three types of thresholding.

        Thresholding converts grayscale to pure black/white (binary).
        Every pixel becomes either 0 or 255.

        Simple binary   → one global cutoff value
        Otsu            → auto-calculates optimal cutoff from histogram
        Adaptive        → different cutoff for different regions of image
        """
        gray = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2GRAY)

        # SIMPLE BINARY: one global cutoff
        _, simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # OTSU'S METHOD: auto-calculates optimal cutoff from histogram
        otsu_thresh, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"Otsu automatically chose threshold: {otsu_thresh:.1f}")

        # ADAPTIVE: different cutoff per image region
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,  # must be odd
            C=2
        )

        self._plot_thresholding(gray, simple, otsu, adaptive, otsu_thresh)

    def _plot_thresholding(self, gray, simple, otsu, adaptive, otsu_thresh):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle('Day 3: Thresholding Comparison', fontsize=16)

        images = [gray, simple, otsu, adaptive]
        titles = [
            'Original Grayscale',
            'Simple Binary\n(threshold=127)',
            f'Otsu Auto\n(threshold={otsu_thresh:.0f}, calculated)',
            'Adaptive\n(per-region threshold)'
        ]

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('day3_threshold.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: day3_threshold.png")

    # ------------------------------------------------------------------
    # DAY 4
    # ------------------------------------------------------------------

    def morphological_operations(self):
        """
        Morphological operations reshape the white regions in a binary/edge image.
        We feed them the Canny output from Day 3 and transform it four different ways
        to understand what each operation does to stock chart edge lines.

        The pipeline here is: grayscale → bilateral filter → Canny → morph ops.
        We rerun bilateral+Canny internally instead of accepting them as params
        so this method is self-contained and callable from find_candle_contours too.
        """
        # first convert to grayscale — all our processing lives in single-channel space
        gray = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2GRAY)

        # bilateral filter before Canny — we validated this is the best filter for
        # chart images on Day 2 because it smooths noise while keeping candle edges sharp
        bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # run Canny on the bilateral output — 50/150 is our "medium" setting from Day 3
        # these thresholds hit the sweet spot between catching real edges and avoiding noise
        canny = cv2.Canny(bilateral, 50, 150)

        # define the structuring element — this is the "brush" morph ops use to probe the image
        # MORPH_RECT = flat square kernel (as opposed to MORPH_ELLIPSE which is round)
        # 3x3 is small enough to be precise but large enough to actually have an effect
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def _plot_morphological(self, canny, eroded, dilated, opened, closed):
        """
        Private helper — just handles the matplotlib side of morphological_operations.
        Kept separate so the main method stays focused on the actual CV logic.
        """
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle('Day 4: Morphological Operations', fontsize=16)

        images = [canny, eroded, dilated, opened, closed]
        titles = [
            'Canny Input',
            'Erosion\n(shrinks white)',
            'Dilation\n(expands white)',
            'Opening\n(removes noise)',
            'Closing\n(fills gaps) ← our choice'
        ]

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('day4_morphological.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: day4_morphological.png")


# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/sanjanawaghray/Documents/projects/Computer-Vision-1/learning/data/raw/chart1.webp"

    processor = ChartPreprocessor(image_path)

    # Day 1 — done, comment out
    # processor.explore_as_array()
    # processor.visualize_all()

    # Day 2 — done, comment out
    # processor.apply_filters()
    # processor.compare_filter_edges()

    # Day 3 — running today
    processor.canny_deep_dive()
    processor.thresholding()