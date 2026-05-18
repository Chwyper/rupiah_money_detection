# --- Konstanta Sistem Skoring ---
WEIGHT_SATURATION = 0.5    # Bobot kelompok fitur saturasi warna
WEIGHT_WATERMARK = 0.5     # Bobot kelompok fitur watermark

AUTHENTIC_THRESHOLD = 35   # Skor gabungan ≥ ini → label ASLI
WARNING_THRESHOLD = 50     # Skor gabungan < ini → tampilkan pesan warning


def compute_final_result(
    saturation_result: dict,
    watermark_result: dict,
    image_metadata: dict,
    total_time_ms: float,
) -> dict:
    """
    Gabungkan hasil dari saturation_checker dan watermark_checker
    menjadi satu response API yang lengkap.

    Langkah:
        1. Ambil skor masing-masing kelompok fitur.
        2. Hitung skor gabungan berbobot.
        3. Tentukan label ASLI / PALSU dan nilai is_authentic.
        4. Susun pesan warning jika confidence rendah.
        5. Rakit dictionary response akhir.

    Args:
        saturation_result : Dict output dari check_saturation().
        watermark_result  : Dict output dari check_watermark().
        image_metadata    : Dict output dari _extract_metadata() di preprocessor.
        total_time_ms     : Total waktu pemrosesan end-to-end dalam milidetik.

    Returns:
        Dictionary lengkap sesuai format output API yang disepakati.
    """
    # --- Langkah 1: Ambil skor tiap kelompok ---
    saturation_score = saturation_result["score"]
    watermark_score = watermark_result["score"]

    # --- Langkah 2: Hitung skor gabungan berbobot ---
    combined_score = (
        saturation_score * WEIGHT_SATURATION
        + watermark_score * WEIGHT_WATERMARK
    )
    confidence = int(round(combined_score))

    # --- Langkah 3: Tentukan label dan is_authentic ---
    is_authentic = confidence >= AUTHENTIC_THRESHOLD
    label = "ASLI" if is_authentic else "PALSU"

    # --- Langkah 4: Susun pesan warning ---
    warning = _build_warning(
        confidence=confidence,
        is_authentic=is_authentic,
        saturation_score=saturation_score,
        watermark_score=watermark_score,
    )

    # --- Langkah 5: Rakit response ---
    return {
        "label": label,
        "confidence": confidence,
        "is_authentic": is_authentic,
        "warning": warning,
        "detail": {
            "saturation": {
                "score": saturation_result["score"],
                "mean_value": saturation_result["mean_value"],
                "threshold_used": saturation_result["threshold_used"],
                "verdict": saturation_result["verdict"],
            },
            "watermark": {
                "score": watermark_result["score"],
                "otsu_threshold": watermark_result["otsu_threshold"],
                "white_pixel_ratio": watermark_result["white_pixel_ratio"],
                "variance": watermark_result["variance"],
                "contours_found": watermark_result["contours_found"],
                "verdict": watermark_result["verdict"],
            },
        },
        "image_metadata": image_metadata,
        "processing_time_ms": round(total_time_ms, 2),
    }


def _build_warning(
    confidence: int,
    is_authentic: bool,
    saturation_score: int,
    watermark_score: int,
) -> str | None:
    """
    Susun pesan warning yang informatif berdasarkan kondisi hasil analisis.

    Warning ditampilkan dalam dua situasi:
        1. Confidence rendah (< WARNING_THRESHOLD) — hasil tidak meyakinkan.
        2. Dua kelompok fitur memberikan sinyal yang saling bertentangan
           (satu tinggi, satu rendah) — gambar ambigu.

    Args:
        confidence      : Skor gabungan akhir (0–100).
        is_authentic    : True jika label ASLI.
        saturation_score: Skor dari saturation_checker (0–100).
        watermark_score : Skor dari watermark_checker (0–100).

    Returns:
        String pesan warning, atau None jika tidak ada peringatan.
    """
    warnings = []

    # Kondisi 1: Confidence rendah
    if confidence < WARNING_THRESHOLD:
        if is_authentic:
            warnings.append(
                f"Hasil menunjukkan ASLI namun confidence rendah ({confidence}/100). "
                "Disarankan untuk memverifikasi secara manual."
            )
        else:
            warnings.append(
                f"Hasil menunjukkan PALSU namun confidence rendah ({confidence}/100). "
                "Disarankan untuk memverifikasi secara manual."
            )

    # Kondisi 2: Sinyal dua kelompok fitur bertentangan
    score_gap = abs(saturation_score - watermark_score)
    if score_gap >= 40:
        warnings.append(
            f"Terdapat perbedaan signifikan antara skor saturasi ({saturation_score}) "
            f"dan skor watermark ({watermark_score}). "
            "Kualitas foto mungkin mempengaruhi hasil — coba foto ulang dengan pencahayaan merata."
        )

    if not warnings:
        return None

    # Gabungkan semua pesan warning jika lebih dari satu
    return " | ".join(warnings)