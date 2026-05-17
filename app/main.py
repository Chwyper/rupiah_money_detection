import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.services.preprocessor import validate_and_preprocess
from app.services.saturation_checker import check_saturation
from app.services.watermark_checker import check_watermark
from app.services.scorer import compute_final_result


# ---------------------------------------------------------------------------
# Lifespan — startup & shutdown (pengganti @app.on_event yang sudah deprecated)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Blok ini dijalankan satu kali saat server start (sebelum yield)
    dan satu kali saat server shutdown (setelah yield).
    Karena proyek ini tidak menggunakan database atau resource eksternal,
    lifespan dipakai sebagai penanda saja agar pola sudah benar sejak awal.
    """
    print("✅ Money Detector API siap menerima request.")
    yield
    print("🛑 Money Detector API dimatikan.")


# ---------------------------------------------------------------------------
# Inisialisasi Aplikasi
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Money Detector API",
    description=(
        "Backend API deteksi keaslian uang kertas Rupiah (Rp 50.000 & Rp 100.000) "
        "menggunakan analisis saturasi warna dan Otsu thresholding. "
        "Rule-based, tanpa machine learning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Frontend boleh dari origin mana pun
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoint: GET /
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root():
    """
    Informasi dasar service.
    Berguna untuk memastikan API sudah berjalan setelah deploy.
    """
    return {
        "service": "Money Detector API",
        "version": "1.0.0",
        "description": "Deteksi keaslian uang kertas Rupiah Rp 50.000 & Rp 100.000.",
        "docs": "/docs",
        "health": "/health",
        "analyze": "/analyze",
    }


# ---------------------------------------------------------------------------
# Endpoint: GET /health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Info"])
def health_check():
    """
    Status server — dipakai oleh Render.com untuk health monitoring.
    Selalu mengembalikan status 'ok' selama server berjalan normal.
    """
    return {
        "status": "ok",
        "message": "Server berjalan normal.",
    }


# ---------------------------------------------------------------------------
# Endpoint: POST /analyze
# ---------------------------------------------------------------------------

@app.post("/analyze", tags=["Detection"])
async def analyze(file: UploadFile = File(...)):
    """
    Endpoint utama — terima gambar uang, kembalikan hasil deteksi keaslian.

    **Input:**
    - `file`: File gambar (JPG / PNG / WEBP), maksimal 10 MB.
      Gambar harus sudah di-crop ke area uang oleh frontend.

    **Output:**
    JSON dengan label ASLI/PALSU, confidence, detail per kelompok fitur,
    metadata gambar, dan waktu pemrosesan.

    **Error yang mungkin dikembalikan:**
    - `400 Bad Request`: Format tidak didukung, ukuran melebihi batas,
      atau file bukan gambar valid.
    - `500 Internal Server Error`: Kesalahan tak terduga di sisi server.
    """
    start_time = time.perf_counter()

    # --- Langkah 1: Baca bytes dari file upload ---
    try:
        file_bytes = await file.read()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Gagal membaca file yang diupload. Pastikan request valid.",
        )

    # --- Langkah 2: Validasi dan preprocessing ---
    try:
        prep_result = validate_and_preprocess(
            file_bytes=file_bytes,
            content_type=file.content_type or "",
        )
    except ValueError as e:
        # ValueError dari preprocessor → error input pengguna → 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat memproses gambar.",
        )

    image = prep_result["image"]
    metadata = prep_result["metadata"]

    # --- Langkah 3: Analisis saturasi warna ---
    try:
        saturation_result = check_saturation(image)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan pada analisis saturasi.",
        )

    # --- Langkah 4: Analisis watermark ---
    try:
        watermark_result = check_watermark(image)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan pada analisis watermark.",
        )

    # --- Langkah 5: Hitung skor akhir dan rakit response ---
    total_time_ms = (time.perf_counter() - start_time) * 1000

    try:
        response = compute_final_result(
            saturation_result=saturation_result,
            watermark_result=watermark_result,
            image_metadata=metadata,
            total_time_ms=total_time_ms,
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat menyusun hasil analisis.",
        )

    return response