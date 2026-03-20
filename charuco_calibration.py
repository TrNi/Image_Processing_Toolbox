import cv2
import numpy as np
import pickle
from reportlab.lib.pagesizes import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io

# =========================
# Board configuration
# =========================
squares_x = 12
squares_y = 16

square_size_mm = 45.0
marker_size_mm = 0.7 * square_size_mm

dictionary_id = cv2.aruco.DICT_4X4_100

# Convert to meters for OpenCV (recommended convention)
square_size_m = square_size_mm / 1000.0
marker_size_m = marker_size_mm / 1000.0

# =========================
# Create dictionary & board
# =========================
aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)

board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    squareLength=square_size_m,
    markerLength=marker_size_m,
    dictionary=aruco_dict
)

# =========================
# Render board at high resolution
# =========================
# Target print resolution: 600 DPI
dpi = 400

board_width_mm = squares_x * square_size_mm
board_height_mm = squares_y * square_size_mm

px_per_mm = dpi / 25.4

img_width_px = int(board_width_mm * px_per_mm)
img_height_px = int(board_height_mm * px_per_mm)

board_img = board.generateImage(
    outSize=(img_width_px, img_height_px),
    marginSize=int(10 * px_per_mm),  # 10 mm margin
    borderBits=1
)

# =========================
# Save as high-quality PDF
# =========================
pdf_filename = f"charuco_{squares_x}x{squares_y}_{round(square_size_mm)}mm.pdf"

c = canvas.Canvas(
    pdf_filename,
    pagesize=(board_width_mm * mm, board_height_mm * mm)
)

# Convert OpenCV image → PDF image
img_rgb = cv2.cvtColor(board_img, cv2.COLOR_GRAY2RGB)
img_pil = ImageReader(io.BytesIO(
    cv2.imencode(".png", img_rgb)[1].tobytes()
))

c.drawImage(
    img_pil,
    x=0,
    y=0,
    width=board_width_mm * mm,
    height=board_height_mm * mm,
    preserveAspectRatio=True,
    mask="auto"
)

c.showPage()
c.save()

print(f"Saved ChArUco board PDF: {pdf_filename}")

# =========================
# Save parameters & dictionary info
# =========================
params = {
    "board_type": "charuco",
    "squares_x": squares_x,
    "squares_y": squares_y,
    "square_size_mm": square_size_mm,
    "marker_size_mm": marker_size_mm,
    "square_size_m": square_size_m,
    "marker_size_m": marker_size_m,
    "dictionary_id": dictionary_id,
    "dictionary_name": "DICT_4X4_100",
    "dpi": dpi,
    "image_resolution_px": (img_width_px, img_height_px)
}

with open(f"charuco_{squares_x}x{squares_y}_{round(square_size_mm)}mm_params.pkl", "wb") as f:
    pickle.dump(params, f)

print(f"Saved board parameters to {pdf_filename.replace('.pdf', '_params.pkl')}")

