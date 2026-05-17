import cv2
import numpy as np

# --- Konstanta Filter Kontur ---
# Luas area minimum piksel agar kontur dianggap watermark (bukan noise)
CONTOUR_MIN_AREA = 150
# Luas area maksimum piksel — buang kontur yang terlalu besar (bukan detail watermark)
CONTOUR_MAX_AREA = 40_000

# --- Konstanta Skoring Between-Class Variance ---
# Variance minimum agar mendapat skor penuh dari komponen ini
VARIANCE_PERFECT = 3000.0
# Variance di bawah ini dianggap kontras sangat rendah
VARIANCE_FLOOR = 200.0

# --- Bobot Komponen Skor Watermark ---
# Skor akhir watermark = gabungan tiga komponen berikut
WEIGHT_VARIANCE = 0.5      # Between-class variance → kontras dominan
WEIGHT_WHITE_RATIO = 0.25  # Rasio piksel putih → keseimbangan tonal
WEIGHT_CONTOURS = 0.25     # Jumlah kontur terfilter → keberadaan detail


def check_watermark(image_bgr: np.ndarray) -> dict:
    """
    Analisis kelompok fitur 2: Otsu Thresholding + Deteksi Watermark.

    Uang asli Rupiah memiliki kontras cetak yang tinggi dan detail watermark
    yang dapat dideteksi sebagai kontur signifikan setelah binarisasi Otsu.
    Uang palsu cenderung memiliki kontras rendah dan watermark yang kabur.

    Langkah:
        1. Konversi BGR → Grayscale.
        2. Terapkan Otsu Thresholding → gambar biner hitam-putih.
        3. Hitung nilai T (threshold), rasio piksel putih, between-class variance.
        4. Jalankan findContours, filter berdasarkan luas area.
        5. Petakan ketiga fitur ke skor gabungan 0–100.
        6. Tentukan verdict tekstual.

    Args:
        image_bgr: numpy array BGR ukuran 400×400 (hasil preprocessor).

    Returns:
        Dictionary berisi:
            - 'score'             : int, skor watermark 0–100
            - 'otsu_threshold'    : int, nilai T hasil Otsu
            - 'white_pixel_ratio' : float, proporsi piksel putih (0.0–1.0)
            - 'variance'          : float, between-class variance Otsu
            - 'contours_found'    : int, jumlah kontur yang lolos filter
            - 'verdict'           : str, "TERDETEKSI" / "LEMAH" / "TIDAK TERDETEKSI"
    """
    # --- Langkah 1: Konversi ke Grayscale ---
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # --- Langkah 2: Otsu Thresholding ---
    otsu_thresh_value, binary_image = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --- Langkah 3: Ekstrak fitur statistik ---
    white_pixel_ratio = _compute_white_pixel_ratio(binary_image)
    variance = _compute_between_class_variance(gray, int(otsu_thresh_value))

    # --- Langkah 4: Deteksi dan filter kontur ---
    contours_found = _count_valid_contours(binary_image)

    # --- Langkah 5: Hitung skor gabungan ---
    score = _compute_score(variance, white_pixel_ratio, contours_found)

    # --- Langkah 6: Tentukan verdict ---
    verdict = _determine_verdict(score)

    return {
        "score": score,
        "otsu_threshold": int(otsu_thresh_value),
        "white_pixel_ratio": round(white_pixel_ratio, 4),
        "variance": round(variance, 2),
        "contours_found": contours_found,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _compute_white_pixel_ratio(binary_image: np.ndarray) -> float:
    """
    Hitung proporsi piksel putih (nilai 255) terhadap total piksel.

    Uang asli setelah Otsu cenderung menghasilkan rasio yang seimbang
    karena distribusi tonal yang baik antara latar dan detail cetak.

    Args:
        binary_image: Gambar biner hasil Otsu (nilai 0 atau 255).

    Returns:
        Float antara 0.0 (semua hitam) hingga 1.0 (semua putih).
    """
    total_pixels = binary_image.size
    white_pixels = int(np.sum(binary_image == 255))
    return white_pixels / total_pixels


def _compute_between_class_variance(
    gray: np.ndarray,
    threshold: int
) -> float:
    """
    Hitung between-class variance berdasarkan formula Otsu secara manual.

    Between-class variance mengukur seberapa baik threshold T memisahkan
    dua kelas piksel (latar belakang vs objek). Nilai tinggi menandakan
    kontras cetak yang jelas — ciri khas uang asli.

    Formula:
        σ²_B = w0 * w1 * (μ0 - μ1)²

    di mana:
        w0, w1 = proporsi piksel kelas background dan foreground
        μ0, μ1 = rata-rata intensitas masing-masing kelas

    Args:
        gray     : Gambar grayscale numpy array.
        threshold: Nilai T hasil Otsu (integer 0–255).

    Returns:
        Float nilai between-class variance. Kembalikan 0.0 jika salah satu
        kelas kosong (gambar seragam total).
    """
    pixels = gray.flatten().astype(np.float64)
    total = len(pixels)

    background = pixels[pixels <= threshold]
    foreground = pixels[pixels > threshold]

    # Cegah division by zero jika gambar seragam
    if len(background) == 0 or len(foreground) == 0:
        return 0.0

    w0 = len(background) / total
    w1 = len(foreground) / total
    mu0 = float(np.mean(background))
    mu1 = float(np.mean(foreground))

    variance = w0 * w1 * ((mu0 - mu1) ** 2)
    return variance


def _count_valid_contours(binary_image: np.ndarray) -> int:
    """
    Deteksi kontur pada gambar biner dan filter berdasarkan luas area.

    Kontur yang terlalu kecil dianggap noise cetak biasa.
    Kontur yang terlalu besar dianggap bukan detail watermark melainkan
    area latar belakang besar yang tidak informatif.

    Args:
        binary_image: Gambar biner hasil Otsu.

    Returns:
        Integer jumlah kontur yang lolos filter [CONTOUR_MIN_AREA, CONTOUR_MAX_AREA].
    """
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,   # Hanya kontur terluar
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if CONTOUR_MIN_AREA <= area <= CONTOUR_MAX_AREA:
            valid_count += 1

    return valid_count


def _compute_score(
    variance: float,
    white_pixel_ratio: float,
    contours_found: int
) -> int:
    """
    Gabungkan tiga komponen fitur menjadi skor watermark 0–100.

    Komponen dan bobotnya:
        - Between-class variance (50%): indikator kontras utama
        - White pixel ratio      (25%): keseimbangan distribusi tonal
        - Jumlah kontur valid    (25%): keberadaan detail watermark

    Args:
        variance          : Nilai between-class variance.
        white_pixel_ratio : Proporsi piksel putih (0.0–1.0).
        contours_found    : Jumlah kontur yang lolos filter.

    Returns:
        Skor integer dalam rentang [0, 100].
    """
    # Komponen 1: Variance → skor 0–100
    variance_score = _normalize(variance, VARIANCE_FLOOR, VARIANCE_PERFECT)

    # Komponen 2: White pixel ratio → skor terbaik di sekitar 0.3–0.7
    # Di luar rentang itu (terlalu terang atau terlalu gelap) → skor turun
    ratio_score = _score_white_ratio(white_pixel_ratio)

    # Komponen 3: Jumlah kontur → skor terbaik di rentang 5–25 kontur
    contour_score = _score_contours(contours_found)

    # Gabungkan dengan bobot
    combined = (
        variance_score  * WEIGHT_VARIANCE
        + ratio_score   * WEIGHT_WHITE_RATIO
        + contour_score * WEIGHT_CONTOURS
    )

    return int(round(min(max(combined, 0), 100)))


def _normalize(value: float, floor: float, perfect: float) -> float:
    """Interpolasi linear nilai ke rentang 0–100 dengan clamp."""
    if value <= floor:
        return 0.0
    if value >= perfect:
        return 100.0
    return (value - floor) / (perfect - floor) * 100.0


def _score_white_ratio(ratio: float) -> float:
    """
    Skor white pixel ratio berbentuk kurva segitiga (triangle function).

    Puncak skor (100) ada di rasio 0.5 — distribusi piksel seimbang.
    Skor turun linear menuju 0 di rasio 0.0 dan 1.0.
    """
    # Jarak dari titik ideal 0.5, maksimal 0.5
    distance = abs(ratio - 0.5)
    score = (1.0 - (distance / 0.5)) * 100.0
    return max(score, 0.0)


def _score_contours(contours_found: int) -> float:
    """
    Skor jumlah kontur valid.

    Skala:
        - 0 kontur      → skor 0   (tidak ada detail terdeteksi)
        - 5–25 kontur   → skor 100 (rentang ideal watermark asli)
        - > 50 kontur   → skor turun (terlalu bising, kemungkinan noise)
    """
    if contours_found == 0:
        return 0.0
    if 5 <= contours_found <= 25:
        return 100.0
    if contours_found < 5:
        return _normalize(float(contours_found), 0, 5)
    # contours_found > 25: turun linear hingga 0 di 50 kontur
    return _normalize(float(50 - contours_found), 0, 25)


def _determine_verdict(score: int) -> str:
    """
    Tentukan verdict tekstual berdasarkan skor watermark akhir.

    Skala:
        - ≥ 60 → "TERDETEKSI"
        - ≥ 35 → "LEMAH"
        - < 35 → "TIDAK TERDETEKSI"
    """
    if score >= 60:
        return "TERDETEKSI"
    elif score >= 35:
        return "LEMAH"
    else:
        return "TIDAK TERDETEKSI"