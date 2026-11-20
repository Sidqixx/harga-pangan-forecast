# %% [markdown]
# # Modul 7: Menjalankan model di dunia nyata (MLOps)
#
# Model yang bagus di backtest baru setengah pekerjaan. Setengahnya lagi: **memastikan ia tetap jalan dan
# tetap benar setiap hari, tanpa ada yang menjaganya**. Itu yang dimaksud rekruter dengan "ngerti production".
#
# 1. Pipeline harian, langkah demi langkah
# 2. Delayed labels: menilai tebakan yang jawabannya baru datang belakangan
# 3. Gerbang rilis: ujian otomatis sebelum model baru dipakai
# 4. Kalau ada yang rusak: skenario dan respons
# 5. Model card
#
# Kode proyek yang dibahas: `.github/workflows/daily.yml`, `live.py`, `MODEL_CARD.md`.

# %%
import json
import os
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "quality.py").exists())
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd

import live
import quality

# %% [markdown]
# ## 1. Pipeline harian
#
# Setiap hari jam 10:17 WIB, GitHub Actions menjalankan `.github/workflows/daily.yml`:

# %%
print(Path(".github/workflows/daily.yml").read_text(encoding="utf-8"))

# %% [markdown]
# Urutannya:
#
# | Langkah | Kalau gagal |
# |---|---|
# | `pytest`: 9 tes | Job berhenti. Kode rusak tidak boleh menyentuh data |
# | `fetch.py`: harga baru + 14 hari terakhir | Chunk yang gagal dilewati, sisanya tetap disimpan, job merah di akhir |
# | `quality.py`: cek kualitas | Kalau data terbaru lebih tua dari 10 hari: job merah, email |
# | `live.py`: gerbang rilis, forecast, peringatan | Gerbang gagal: tidak ada yang dipublikasikan, hasil terakhir yang lolos tetap tampil |
# | commit | Hanya kalau ada perubahan. `if: always()` = tetap jalan walau langkah sebelumnya gagal, supaya data yang sudah berhasil ditarik tidak hilang |
# | `report.py` + deploy | Dashboard dibangun ulang dari apa pun yang ada di repo |
#
# Prinsip yang dipakai di semua langkah:
# - **Idempotent**: dijalankan dua kali hasilnya sama (modul 1). Cron yang terlambat atau terulang tidak merusak apa pun.
# - **Gagal dengan jelas**: job merah + email, bukan diam-diam menghasilkan angka jelek.
# - **Deterministik**: `seed=0`, jadi hasil di laptop sama persis dengan di server GitHub.
#
# Catatan cron: jadwal ditulis dalam **UTC**. `17 3 * * *` = 03:17 UTC = 10:17 WIB. Menit 0 dihindari karena
# GitHub sering menunda atau membuang jadwal di awal jam yang ramai (ini benar-benar terjadi di proyek ini).

# %% [markdown]
# ## 2. Delayed labels
#
# Tebakan untuk 14 hari ke depan baru bisa dinilai setelah harganya keluar. Jadi setiap hari, sebagian tebakan
# lama "matang" dan langsung dinilai. Ayo lihat berapa yang sudah matang.

# %%
df = quality.clean(quality.load_raw())
forecasts = live.load(live.FORECASTS)
scored = live.score(df)
print(f"Forecast tersimpan: {len(forecasts):,} dari {forecasts['origin'].nunique()} origin")
print(f"Sudah dinilai (harga aslinya sudah keluar): {len(scored):,}")
if len(scored):
    print(scored.groupby("h").size().rename("dinilai per horizon").to_string())

# %% [markdown]
# Horizon 1 hari matang paling cepat (besoknya), horizon 14 hari paling lambat. Jadi **sinyal pertama bahwa model
# bermasalah datang dalam sehari**, dari horizon pendek, walau penilaian lengkap butuh 2 minggu.
#
# Yang dipantau di dashboard:
# - error model vs error naive **di hari-hari yang sama** (naive juga disimpan, jadi selalu ada pembanding yang adil),
# - coverage P10–P90 (harus sekitar 80%),
# - untuk peringatan: berapa persen yang benar-benar melonjak setelah 14 hari.

# %% [markdown]
# ## 3. Gerbang rilis
#
# Model dilatih ulang setiap hari. Tapi bagaimana kalau data hari ini aneh, atau ada yang mengubah kode dan
# tidak sengaja merusak fitur? Model rusak itu akan langsung tayang.
#
# Gerbang rilis (`live.gate()`) mencegahnya. Sebelum mempublikasikan:
# 1. Latih ulang dengan resep yang sama **seolah-olah 4 minggu lalu**.
# 2. Uji di 4 minggu terakhir, yang tidak pernah dilihat model itu.
# 3. Lolos hanya kalau **ketiganya** terpenuhi.

# %%
status = json.loads(live.STATUS.read_text())
status

# %%
checks = {
    f"Forecast tidak kalah dari naive (≤ naive × {live.NAIVE_TOLERANCE})": status["forecast_mape"] <= status["naive_mape"] * live.NAIVE_TOLERANCE,
    f"Coverage di antara {live.COVERAGE_RANGE[0]:.0%} dan {live.COVERAGE_RANGE[1]:.0%}": live.COVERAGE_RANGE[0] <= status["coverage"] <= live.COVERAGE_RANGE[1],
    "Classifier mengalahkan heuristik": status["spike_ap"] > status["heuristic_ap"],
}
pd.Series(checks).map({True: "lolos", False: "GAGAL"}).to_frame("hasil")

# %% [markdown]
# Kenapa 3 syarat itu:
# - **Kalah dari naive** = model lebih buruk dari "harga terakhir". Tidak ada alasan memakainya.
# - **Coverage di luar 70–90%** = rentang P10–P90 tidak bisa dipercaya lagi.
# - **Kalah dari heuristik** = peringatan dini tidak lebih pintar dari aturan sederhana.
#
# Toleransi 2% di syarat pertama: 4 minggu itu sampel kecil, jadi seri tipis dengan naive dianggap noise, bukan kerusakan.
#
# Istilah industri yang mirip: **champion/challenger**. Model baru (challenger) hanya menggantikan yang lama
# (champion) kalau menang di ujian yang sama. Di proyek ini, pembandingnya adalah naive dan heuristik.

# %% [markdown]
# ## 4. Kalau ada yang rusak
#
# | Kejadian | Yang terjadi otomatis | Yang kamu lakukan |
# |---|---|---|
# | Situs PIHPS mati 3 hari | Fetch gagal, job merah, email. Data lama aman. Saat hidup lagi, jendela 14 hari menambal bolongnya | Tidak perlu apa-apa, kecuali lebih dari 10 hari (quality gagal) |
# | Format data PIHPS berubah | `parse()` gagal atau quality menemukan banyak outlier, job merah | Perbaiki `fetch.py`, tambah tes untuk format baru |
# | Kode diubah dan fiturnya jadi bocor | `tests/test_spike.py` gagal, job berhenti sebelum menyentuh data | Perbaiki fiturnya |
# | Pola harga berubah (drift), model jadi jelek | Gerbang gagal, hasil terakhir yang lolos tetap tampil, email | Cek fitur dan data 4 minggu terakhir, latih ulang dengan perubahan, backtest lagi |
# | Coverage live turun jauh di bawah 80% berminggu-minggu | Terlihat di dashboard | Cari sebabnya (data hari berjalan belum lengkap? drift?) |
# | Jadwal cron ditunda GitHub | Run telat atau terlewat | Picu manual dari tab Actions; idempotent, jadi aman |

# %% [markdown]
# ## 5. Model card
#
# `MODEL_CARD.md` adalah "label kemasan" model. Bagian yang paling sering ditanyakan interviewer:
#
# - **Penggunaan yang dimaksud, dan yang bukan.** Model ini membantu keputusan stok, bukan statistik resmi.
# - **Evaluasi**: angka backtest dengan pembandingnya.
# - **Keterbatasan**: 1 dari 3 peringatan tidak terjadi; lonjakan mendadak tanpa tanda (banjir, kebijakan) sering lolos;
#   rentang cabai 8–14 hari sedikit kesempitan.
# - **Risiko**: peringatan publik bisa memicu penimbunan.
#
# Menulis keterbatasan dengan jujur bukan tanda kelemahan. Justru itu yang membedakan proyek serius dari demo.

# %%
print(Path("MODEL_CARD.md").read_text(encoding="utf-8")[:1500])

# %% [markdown]
# ## Latihan
#
# 1. Buka tab **Actions** di repo GitHub. Cari satu run dengan event `schedule` dan satu dengan `workflow_dispatch`.
#    Apa bedanya?
# 2. Di bagian 2, kenapa horizon 14 hari selalu paling sedikit dinilai?
# 3. Bayangkan coverage di gerbang hari ini 0,95. Gerbang lolos atau gagal? Apa artinya bagi user?
# 4. Interviewer bertanya: "Model kamu tiba-tiba jelek minggu ini. Bagaimana kamu tahu, dan apa yang terjadi
#    pada user?" Jawab dalam 3 kalimat.

# %%
# Kerjakan di sini


# %% [markdown]
# ## Jawaban
#
# 1. `schedule` = dijalankan otomatis oleh jadwal cron; `workflow_dispatch` = dipicu manual (tombol Run workflow).
# 2. Karena tebakan h14 baru matang 14 hari setelah dibuat; tebakan h1 matang besoknya.
# 3. Gagal (batas atas 90%). Rentangnya kelebaran: masih benar, tapi tidak informatif. User tetap melihat
#    forecast terakhir yang lolos, dan saya mendapat email.
# 4. "Gerbang rilis harian menguji model baru di 4 minggu yang belum ia lihat; kalau kalah dari naive atau
#    rentangnya tidak jujur, model itu tidak dipublikasikan dan saya mendapat email. User tetap melihat versi
#    terakhir yang lolos. Lalu saya cek data dan fitur 4 minggu terakhir untuk mencari penyebabnya."
