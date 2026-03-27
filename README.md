# AutoVision Pipeline

A computer vision preprocessing pipeline built from scratch to process raw stock chart screenshots into clean, structured data ready for machine learning. Each day of development adds a new layer to the pipeline — from basic image loading to edge detection, morphological operations, and contour-based candle detection.

## Tech Stack

- Python
- OpenCV
- NumPy
- Matplotlib
- PyTorch (coming soon)

## Pipeline Overview

1. **Load & explore** — read image as NumPy array, convert color spaces (RGB, HSV, LAB, grayscale)
2. **Filter** — bilateral filter to reduce noise while keeping candle edges sharp
3. **Edge detection** — Canny with gradient analysis (Sobel X/Y, NMS, hysteresis)
4. **Morphological operations** — closing fills broken candle edges; erosion/opening remove noise
5. **Contour detection** — find and filter candle regions by area, draw bounding boxes and centroids

## What Each Day Builds

- **Day 1** — Image as array: color spaces, resizing, center crop
- **Day 2** — Filtering: Gaussian vs Median vs Bilateral; visual comparison on chart edges
- **Day 3** — Canny deep dive: Sobel gradients, thresholding (simple, Otsu, adaptive)
- **Day 4** — Morphological ops, contour detection, Harris corners, chart region isolation

## How to Run

```bash
# set up environment
python -m venv venv
source venv/bin/activate
pip install opencv-python numpy matplotlib

# run the full preprocessing pipeline
cd learning/autovision-pipeline
python preprocessing.py /path/to/your/chart.webp

# run unit tests
pytest test_preprocessing.py -v
```

## Sample Outputs

Running the pipeline generates these images in the working directory:

| File | What it shows |
|---|---|
| `day1_output.png` | Color space comparison (RGB, HSV, LAB, grayscale) |
| `day2_filters.png` | Gaussian vs Median vs Bilateral filter side-by-side |
| `day2_filter_edges.png` | Canny output after each filter (shows bilateral wins) |
| `day3_canny.png` | Every step inside Canny: blur → Sobel → gradient → edges |
| `day3_threshold.png` | Simple vs Otsu vs adaptive thresholding |
| `day4_morphological.png` | Erosion, dilation, opening, closing on Canny output |
| `day4_contours.png` | 4-panel: original → Canny → closing → detected regions |
| `day4_harris.png` | Harris corner response heatmap and detected corners |
| `day4_isolated.png` | Original vs cropped chart region |
| `pipeline_output.png` | Full 5-step pipeline summary in one image |

## Coming Soon

- PyTorch model training on labeled candle regions
- Weights & Biases experiment tracking
- FastAPI inference endpoint
- Vercel frontend for live chart uploads
