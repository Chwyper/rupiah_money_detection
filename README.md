# Money Detector API

Backend API deteksi keaslian uang kertas Rupiah **Rp 50.000** dan **Rp 100.000**
menggunakan pendekatan *rule-based* berbasis analisis citra digital.

Dibangun sebagai tugas kuliah semester 4 — ITENAS Bandung.

---

## Metode Deteksi

API menggunakan dua kelompok fitur dengan bobot masing-masing 50%:

| Kelompok | Fitur | Bobot |
|---|---|---|
| Saturasi Warna | Mean saturasi channel S (HSV) + Otsu pada channel S | 50% |
| Analisis Watermark | Otsu thresholding + between-class variance + deteksi kontur | 50% |

**Sistem Skoring:**
- Skor gabungan ≥ 60 → label **ASLI**
- Skor gabungan < 60 → label **PALSU**
- Skor gabungan < 70 → tampilkan **pesan warning**

---

## Struktur Proyek

```
money-detector/
├── app/
│   ├── __init__.py
│   ├── main.py                    # Entry point FastAPI
│   └── services/
│       ├── __init__.py
│       ├── preprocessor.py        # Validasi, load, resize, CLAHE
│       ├── saturation_checker.py  # Analisis saturasi warna
│       ├── watermark_checker.py   # Analisis watermark + kontur
│       └── scorer.py              # Skoring gabungan & response
├── requirements.txt
└── README.md
```

---

## Instalasi & Menjalankan Secara Lokal

### Prasyarat

- Python 3.10 atau 3.11 (disarankan 3.11 untuk kompatibilitas OpenCV)

### Langkah-langkah

**1. Clone atau download repository ini**

**2. Buat virtual environment**

```bash
python -m venv venv
```

**3. Aktifkan virtual environment**

Windows (PowerShell):

```powershell
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Jalankan server**

```bash
uvicorn app.main:app --reload
```

**6. Buka Swagger UI di browser**

```
http://localhost:8000/docs
```

---

## Endpoint

| Method | Path | Deskripsi |
|---|---|---|
| GET | `/` | Informasi service |
| GET | `/health` | Status server |
| POST | `/analyze` | Analisis keaslian uang |

### POST `/analyze`

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` — file gambar (JPG / PNG / WEBP, maks. 10 MB)
- Gambar harus sudah di-crop ke area uang oleh frontend

**Response Sukses (200):**

```json
{
  "label": "ASLI",
  "confidence": 79,
  "is_authentic": true,
  "warning": null,
  "detail": {
    "saturation": {
      "score": 85,
      "mean_value": 112.4,
      "threshold_used": 58.2,
      "verdict": "TINGGI"
    },
    "watermark": {
      "score": 70,
      "otsu_threshold": 142,
      "white_pixel_ratio": 0.65,
      "variance": 1840.2,
      "contours_found": 8,
      "verdict": "TERDETEKSI"
    }
  },
  "image_metadata": {
    "resolution": "400x400",
    "blur_score": 1243.5,
    "mean_brightness": 162.0
  },
  "processing_time_ms": 38.7
}
```

**Response Error:**

```json
{ "detail": "Pesan error dalam Bahasa Indonesia." }
```

| Kode | Kondisi |
|---|---|
| 400 | Format tidak didukung / ukuran melebihi batas / file korup |
| 500 | Kesalahan internal server |

---

## Deployment ke Render.com

1. Upload seluruh isi folder `money-detector/` ke repository GitHub
2. Login ke [render.com](https://render.com) → **New Web Service**
3. Hubungkan repository GitHub
4. Isi konfigurasi:

| Field | Nilai |
|---|---|
| Runtime | Python |
| Region | Singapore |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |

5. Klik **Deploy** — tunggu hingga status berubah menjadi **Live**
6. Swagger UI tersedia di: `https://nama-service.onrender.com/docs`

---

## Catatan Teknis

- Gambar diproses **di memori RAM** — tidak ada file yang disimpan ke disk
- Tidak menggunakan database, machine learning, atau file model
- CLAHE diterapkan hanya pada channel **V** (Value) di ruang HSV agar
  channel S (Saturation) tidak terdistorsi sebelum analisis
- Seluruh konstanta skoring berada di bagian atas masing-masing file
  service dan dapat disesuaikan tanpa mengubah logika fungsi

---

## Dependencies

```
fastapi
uvicorn[standard]
python-multipart
opencv-python-headless
Pillow
numpy
```
