"""
sudoku_cells.py - Find a sudoku grid in a photo, straighten it, and split it
into a 9x9 array of JPEG-encoded cell images.

Usage:
    from sudoku_cells import extract_cells
    cells = extract_cells("photo.jpg")      # 9x9 list of JPEG bytes
    cells[0][0]                              # top-left cell, as .jpg bytes

Or set INPUT_PATH / OUTPUT_DIR below and run:  python sudoku_cells.py

Requires: pip install opencv-python numpy    (OpenCV >= 5)
"""

import os

import cv2
import numpy as np


# Settings

INPUT_PATH = "../data/sudoku-image-dataset/v2_train/v2_train/image1078.jpg"  # photo used when running this file directly
OUTPUT_DIR = "./cells"            # where cell_r<row>_c<col>.jpg files are written
DEBUG_DIR = "./debug"             # set to a folder name to save debug images

GRID_SIZE = 9                     # cells per row / column
WARPED_SIZE_PX = 900              # side length of the straightened grid image
CELL_MARGIN = 0.0                 # fraction trimmed off each cell edge (e.g. 0.1)
JPEG_QUALITY = 95                 # 0-100

# Preprocessing
BLUR_KERNEL_SIZE = (7, 7)         # must be odd
BLUR_SIGMA = 0                    # 0 = let OpenCV derive it from the kernel size
WHITE = 255                       # value given to "ink" pixels after thresholding
THRESHOLD_BLOCK_SIZE = 11         # neighbourhood (px) each pixel is compared against
THRESHOLD_OFFSET = 2              # how much darker than its neighbourhood counts as ink

# Grid detection
MIN_CELL_HOLES = 20               # a grid encloses 81 cells
PARENT = 3                        # index of the parent field in OpenCV's hierarchy rows
NO_PARENT = -1

# Debug drawing
RED = (0, 0, 255)                 # OpenCV colours are BGR
OUTLINE_THICKNESS = 4
FONT_SCALE = 1.2
TEXT_THICKNESS = 3
LABEL_OFFSET = (8, 30)            # px from each corner to its label


# Grid finding

def _preprocess(gray: np.ndarray) -> np.ndarray:
    """Binary image where grid lines are white, needed for cv2.findContours"""
    blur = cv2.GaussianBlur(src=gray, ksize=BLUR_KERNEL_SIZE, sigmaX=BLUR_SIGMA)
    return cv2.adaptiveThreshold(
        src=blur,
        maxValue=WHITE,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=THRESHOLD_BLOCK_SIZE,
        C=THRESHOLD_OFFSET,
    )


def _find_grid_corners(binary: np.ndarray) -> np.ndarray:
    """Returns the coordinates of the 4 corners (TL, TR, BR, BL) of the sudoku grid.

    The grid is the largest white shape with many holes (its cells). Picking
    the largest shape alone could grab things like the entire piece of paper if
    it was fully visibleor if the sudoku grid isn't the biggest image on the page.
    """
    # https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_contours/py_table_of_contents_contours.html
    contours, hierarchy = cv2.findContours(
        image=binary, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE
    )
    parents = hierarchy[0][:, PARENT]
    hole_counts = np.bincount(parents[parents != NO_PARENT], minlength=len(contours))
    candidates = [
        contour
        for contour, parent, holes in zip(contours, parents, hole_counts)
        if parent == NO_PARENT and holes >= MIN_CELL_HOLES
    ]
    grid = max(candidates, key=cv2.contourArea)
    # The outline's most extreme points along the diagonals are the grid's corners.
    return _order_corners(grid.reshape(-1, 2).astype(np.float32))


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Pick the top-left, top-right, bottom-right, bottom-left points."""
    s = np.sum(pts, axis=1).ravel()         # x + y
    d = np.diff(pts, axis=1).ravel()        # y - x
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


# Main steps

def warp_grid(img: np.ndarray, size: int = WARPED_SIZE_PX, debug_dir: str | None = DEBUG_DIR) -> np.ndarray:
    """Locate the sudoku grid and return a top-down, square, size x size crop."""
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    binary = _preprocess(gray)
    corners = _find_grid_corners(binary)

    last = size - 1
    target = np.array([[0, 0], [last, 0], [last, last], [0, last]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src=corners, dst=target)
    warped = cv2.warpPerspective(src=img, M=matrix, dsize=(size, size), flags=cv2.INTER_CUBIC)

    if debug_dir:
        _save_debug_images(debug_dir, img, binary, corners, warped)
    return warped


def split_cells(warped: np.ndarray, margin: float = CELL_MARGIN) -> list[list[np.ndarray]]:
    """Cut the straightened grid into 81 cell images.

    margin: fraction of each cell to trim from every side
    (e.g. 0.1) to drop grid-line borders. 0 keeps the full cell.
    """
    h, w = (np.array(warped.shape[:2]) / GRID_SIZE).astype(int).tolist()
    cells = []
    for r in range(GRID_SIZE):
        row = []
        y0, y1 = r * h, (r + 1) * h
        for c in range(GRID_SIZE):
            x0, x1 = c * w, (c + 1) * w
            my, mx = int((y1 - y0) * margin), int((x1 - x0) * margin)
            row.append(warped[y0 + my:y1 - my, x0 + mx:x1 - mx].copy())
        cells.append(row)
    return cells


def extract_cells(
    path: str,
    size: int = WARPED_SIZE_PX,
    margin: float = CELL_MARGIN,
    jpeg_quality: int = JPEG_QUALITY,
    debug_dir: str | None = DEBUG_DIR,
) -> list[list[bytes]]:
    """Photo path -> 9x9 list of JPEG bytes.

    Requires photos to be given upright.
    TODO: Make it work with any orientation.
    """
    img = cv2.imread(filename=path)
    warped = warp_grid(img, size=size, debug_dir=debug_dir)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    return [
        [cv2.imencode(ext=".jpg", img=cell, params=params)[1].tobytes() for cell in row]
        for row in split_cells(warped, margin=margin)
    ]


# Debug output

def _save_debug_images(debug_dir, img, binary, corners, warped) -> None:
    """Save the threshold mask, the detected outline, and the straightened grid."""
    os.makedirs(debug_dir, exist_ok=True)
    overlay = img.copy()
    cv2.polylines(img=overlay, pts=[corners.astype(np.int32)], isClosed=True,
                  color=RED, thickness=OUTLINE_THICKNESS)
    for (x, y), label in zip(corners.astype(int), ("TL", "TR", "BR", "BL")):
        cv2.putText(img=overlay, text=label, org=(x + LABEL_OFFSET[0], y + LABEL_OFFSET[1]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE,
                    color=RED, thickness=TEXT_THICKNESS)
    cv2.imwrite(filename=os.path.join(debug_dir, "debug_threshold.jpg"), img=binary)
    cv2.imwrite(filename=os.path.join(debug_dir, "debug_corners.jpg"), img=overlay)
    cv2.imwrite(filename=os.path.join(debug_dir, "debug_warped.jpg"), img=warped)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for r, row in enumerate(extract_cells(INPUT_PATH)):
        for c, jpg in enumerate(row):
            with open(os.path.join(OUTPUT_DIR, f"cell_r{r}_c{c}.jpg"), "wb") as f:
                f.write(jpg)
    print(f"Wrote {GRID_SIZE * GRID_SIZE} cells to {OUTPUT_DIR}/")
