import cv2
import numpy as np

# --- Konstanta Skoring ---
# Batas bawah rata-rata saturasi yang dianggap "vivid" (uang asli)
SATURATION_HIGH_THRESHOLD = 80.0
# Batas minimum agar mendapat skor penuh (100)
SATURATION_PERFECT = 130.0
# Batas bawah absolut — di bawah ini skor 0
SATURATION_FLOOR = 20.0


def check_saturation(image_bgr: np.ndarray) -> dict:
    """
    Analisis kelompok fitur 1: Saturasi Warna.

    Uang asli Rupiah memiliki tinta warna vivid dengan saturasi tinggi.
    Uang palsu cenderung pucat (washed-out) karena keterbatasan proses cetak.

    Langkah:
        1. Konversi BGR → HSV, ambil channel S (Saturation).
        2. Hitung threshold otomatis Otsu pada channel S.
        3. Hitung rata-rata saturasi seluruh gambar.
        4. Petakan rata-rata saturasi ke skor 0–100.
        5. Tentukan verdict tekstual.

    Args:
        image_bgr: numpy array BGR ukuran 400×400 (hasil preprocessor).

    Returns:
        Dictionary berisi:
            - 'score'          : int, skor saturasi 0–100
            - 'mean_value'     : float, rata-rata nilai channel S
            - 'threshold_used' : float, nilai threshold Otsu pada channel S
            - 'verdict'        : str, "TINGGI" / "SEDANG" / "RENDAH"
    """
    # --- Langkah 1: Ekstrak channel S dari ruang HSV ---
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, s_channel, _ = cv2.split(image_hsv)

    # --- Langkah 2: Otsu Thresholding pada channel S ---
    # cv2.threshold dengan flag OTSU mengembalikan (threshold_value, binary_image)
    otsu_threshold, _ = cv2.threshold(
        s_channel,
        0,           # Nilai awal diabaikan saat OTSU aktif
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --- Langkah 3: Rata-rata saturasi ---
    mean_saturation = float(np.mean(s_channel))

    # --- Langkah 4: Konversi ke skor 0–100 ---
    score = _compute_score(mean_saturation)

    # --- Langkah 5: Verdict tekstual ---
    verdict = _determine_verdict(mean_saturation)

    return {
        "score": score,
        "mean_value": round(mean_saturation, 2),
        "threshold_used": round(float(otsu_threshold), 2),
        "verdict": verdict,
    }


def _compute_score(mean_saturation: float) -> int:
    """
    Petakan nilai rata-rata saturasi (0–255) ke skor integer 0–100.

    Kurva linear dengan clamp:
        - mean_saturation ≤ SATURATION_FLOOR   → skor 0
        - mean_saturation ≥ SATURATION_PERFECT  → skor 100
        - Di antara keduanya                    → interpolasi linear

    Args:
        mean_saturation: Rata-rata nilai channel S (0.0–255.0).

    Returns:
        Skor integer dalam rentang [0, 100].
    """
    if mean_saturation <= SATURATION_FLOOR:
        return 0
    if mean_saturation >= SATURATION_PERFECT:
        return 100

    # Interpolasi linear antara FLOOR dan PERFECT
    score_raw = (
        (mean_saturation - SATURATION_FLOOR)
        / (SATURATION_PERFECT - SATURATION_FLOOR)
        * 100
    )
    return int(round(score_raw))


def _determine_verdict(mean_saturation: float) -> str:
    """
    Tentukan verdict tekstual berdasarkan rata-rata saturasi.

    Skala:
        - ≥ SATURATION_HIGH_THRESHOLD → "TINGGI"  (indikasi asli)
        - ≥ SATURATION_FLOOR          → "SEDANG"  (ambigu)
        - < SATURATION_FLOOR          → "RENDAH"  (indikasi palsu)

    Args:
        mean_saturation: Rata-rata nilai channel S.

    Returns:
        String verdict: "TINGGI", "SEDANG", atau "RENDAH".
    """
    if mean_saturation >= SATURATION_HIGH_THRESHOLD:
        return "TINGGI"
    elif mean_saturation >= SATURATION_FLOOR:
        return "SEDANG"
    else:
        return "RENDAH"