# Belajar dari nol

Jalur belajar proyek ini dari dasar. Tiap modul menjelaskan konsepnya dengan bahasa biasa, lalu langsung membuktikannya dengan data harga PIHPS yang ada di repo ini. Tidak perlu hafal rumus; yang penting paham *kenapa*.

## Cara pakai

1. Install tambahan untuk belajar (sekali saja):
   ```powershell
   .venv\Scripts\python.exe -m pip install -r docs\belajar\requirements.txt
   ```
2. Buka file `.py` di VS Code (butuh extension **Python** dan **Jupyter** dari Microsoft, pilih interpreter `.venv`). Tiap blok yang diawali `# %%` adalah satu *cell*: klik **Run Cell** di atasnya (atau `Shift+Enter`). Hasil dan grafik muncul di jendela Interactive, sama seperti Jupyter notebook.
3. Jalankan cell **berurutan dari atas**. Baca teks penjelasannya dulu, tebak hasilnya, baru jalankan.
4. Tiap modul ditutup dengan **Latihan**. Coba kerjakan sendiri di cell kosong, baru cek **Jawaban** di paling bawah.

Kalau ketemu istilah yang bikin bingung, buka [`00_kamus.md`](00_kamus.md).

## Urutan

| Modul | Isi | Kode proyek yang dibahas |
|---|---|---|
| [`01_data.py`](01_data.py) | Bentuk data, format panjang vs lebar, data bolong, cek kualitas, idempotent | `fetch.py`, `quality.py` |
| [`02_baseline_backtest.py`](02_baseline_backtest.py) | Apa itu forecast, horizon, baseline, MAPE, kenapa tidak boleh acak, backtest | `backtest.py` |
| [`03_fitur_leakage.py`](03_fitur_leakage.py) | Log return, volatilitas, mean reversion, fitur Lebaran, leakage | `model.py` (bagian fitur) |
| [`04_tree_ke_lightgbm.py`](04_tree_ke_lightgbm.py) | Decision tree dari nol, boosting dari nol, lalu LightGBM, overfitting | `model.py` (bagian model) |
| [`05_kuantil_coverage.py`](05_kuantil_coverage.py) | Kuantil, P10/P50/P90, pinball loss, coverage | `model.py`, `backtest.py` |
| [`06_klasifikasi.py`](06_klasifikasi.py) | Label lonjakan, kelas tidak seimbang, precision, recall, average precision, cut-off | `spike.py` |
| [`07_mlops.py`](07_mlops.py) | Pipeline harian, delayed labels, gerbang rilis, monitoring, model card | `live.py`, `.github/workflows/daily.yml` |

Setelah semua modul, baca lagi [`../LEARNING.md`](../LEARNING.md): sekarang isinya seharusnya terasa seperti ringkasan, bukan materi baru.
