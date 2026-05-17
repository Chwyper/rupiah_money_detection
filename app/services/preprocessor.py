import io
import time
import cv2
import numpy as np
from PIL import Image

# --- Konstanta ---
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
TARGET_SIZE = (400, 400)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


def validate_and_preprocess(file_bytes: bytes, content_type: str) -> dict:
    """
    Validasi file upload, load ke numpy array BGR,
    terapkan preprocessing, dan ekstrak metadata gambar.

    Args:
        file_bytes : Raw bytes dari file yang diupload.
        content_type: MIME type dari file (misal: 'image/jpeg').

    Returns:
        Dictionary berisi:
            - 'image'    : numpy array BGR hasil preprocessing (400x400)
            - 'metadata' : dict resolusi, blur_score, mean_brightness
            - 'load_time_ms': waktu load + preprocessing dalam milidetik

    Raises:
        ValueError: Jika format tidak didukung, ukuran melebihi batas,
                    atau file tidak dapat dibaca sebagai gambar valid.
    """
    start_time = time.perf_counter()

    # --- Langkah 1: Validasi ukuran file ---
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Ukuran file melebihi batas maksimal 10 MB. "
            f"Ukuran diterima: {len(file_bytes) / (1024 * 1024):.2f} MB."
        )

    # --- Langkah 2: Validasi format via Pillow ---
    try:
        pil_image = Image.open(io.BytesIO(file_bytes))
        pil_image.verify()  # Pastikan file tidak korup
    except Exception:
        raise ValueError("File tidak dapat dibaca. Pastikan file adalah gambar yang valid.")

    # Buka ulang setelah verify() — verify() menutup stream internal
    pil_image = Image.open(io.BytesIO(file_bytes))

    if pil_image.format not in ALLOWED_FORMATS:
        raise ValueError(
            f"Format '{pil_image.format}' tidak didukung. "
            f"Gunakan JPG, PNG, atau WEBP."
        )

    # --- Langkah 3: Konversi Pillow → numpy array BGR ---
    # Pillow membaca dalam mode RGB; OpenCV menggunakan BGR
    pil_image = pil_image.convert("RGB")
    image_rgb = np.array(pil_image, dtype=np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # --- Langkah 4: Resize ke 400×400 ---
    image_resized = cv2.resize(image_bgr, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # --- Langkah 5: CLAHE pada channel V (HSV) ---
    image_preprocessed = _apply_clahe(image_resized)

    # --- Langkah 6: Ekstrak metadata ---
    metadata = _extract_metadata(image_preprocessed)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "image": image_preprocessed,
        "metadata": metadata,
        "load_time_ms": round(elapsed_ms, 2),
    }


def _apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    """
    Normalisasi pencahayaan menggunakan CLAHE pada channel V (Value) di ruang HSV.
    Channel H (Hue) dan S (Saturation) tidak diubah agar warna tetap akurat
    untuk analisis saturasi selanjutnya.

    Args:
        image_bgr: numpy array BGR ukuran 400×400.

    Returns:
        numpy array BGR dengan pencahayaan yang sudah dinormalisasi.
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    v_clahe = clahe.apply(v)

    image_hsv_clahe = cv2.merge([h, s, v_clahe])
    image_bgr_result = cv2.cvtColor(image_hsv_clahe, cv2.COLOR_HSV2BGR)

    return image_bgr_result


def _extract_metadata(image_bgr: np.ndarray) -> dict:
    """
    Ekstrak informasi deskriptif dari gambar hasil preprocessing.

    Metadata yang dihasilkan:
    - resolution   : string "WxH" dari gambar
    - blur_score   : variance of Laplacian — semakin tinggi semakin tajam
    - mean_brightness: rata-rata nilai kecerahan pada channel V (HSV)

    Args:
        image_bgr: numpy array BGR hasil preprocessing.

    Returns:
        Dictionary berisi ketiga metadata di atas.
    """
    h, w = image_bgr.shape[:2]
    resolution = f"{w}x{h}"

    # Blur score: variance of Laplacian pada grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

    # Mean brightness dari channel V pada HSV
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, _, v_channel = cv2.split(image_hsv)
    mean_brightness = round(float(np.mean(v_channel)), 2)

    return {
        "resolution": resolution,
        "blur_score": blur_score,
        "mean_brightness": mean_brightness,
    }