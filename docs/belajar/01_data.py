# %% [markdown]
# # Modul 1: Kenalan dengan datanya
#
# Sebelum ada model apa pun, kita harus paham bentuk datanya. Di modul ini:
# 1. Data mentah itu bentuknya seperti apa
# 2. Format **panjang** vs **lebar**, dan kenapa dua-duanya dipakai
# 3. Hari yang **bolong**, dan cara mengisinya dengan aman
# 4. **Outlier**: membedakan salah ketik dari lonjakan asli
# 5. **Idempotent**: kenapa pipeline harian aman dijalankan berkali-kali
#
# Kode proyek yang dibahas: `fetch.py`, `quality.py`.

# %%
# Setup: cari folder proyek (tempat quality.py berada) supaya modul proyek bisa di-import.
import os
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "quality.py").exists())
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest
import quality

pd.set_option("display.width", 140)

# %% [markdown]
# ## 1. Data mentah
#
# `fetch.py` menyimpan harga ke `data/raw/YYYY-MM.csv`, satu file per bulan.
# `quality.load_raw()` membaca semua file itu jadi satu tabel.

# %%
raw = quality.load_raw()
print(f"{len(raw):,} baris")
raw.head()

# %% [markdown]
# Tiap baris = **satu harga, di satu tanggal, di satu provinsi, untuk satu komoditas**.
# Ini yang disebut **format panjang** (*long format*).
#
# Berapa banyak kombinasi (komoditas × provinsi)? Tiap kombinasi disebut satu **deret**.

# %%
print("Komoditas:", raw["commodity"].nunique())
print("Provinsi (termasuk 'Semua Provinsi' = rata-rata nasional):", raw["province"].nunique())
print("Deret:", raw.groupby(["commodity_id", "province"]).ngroups)
print("Rentang tanggal:", raw["date"].min().date(), "s.d.", raw["date"].max().date())

# %% [markdown]
# ## 2. Format panjang vs lebar
#
# **Panjang** enak untuk disimpan dan di-filter (tinggal tambah baris).
# **Lebar** enak untuk hitungan deret waktu: satu kolom = satu deret, jadi
# "harga 7 hari lalu" tinggal geser kolomnya ke bawah 7 baris.
#
# `backtest.to_wide()` mengubah panjang jadi lebar, dengan **satu baris per hari kalender**
# (termasuk Sabtu–Minggu, yang isinya kosong).

# %%
wide = backtest.to_wide(quality.clean(raw))
print(wide.shape, "= (hari, deret)")
dki = wide[("com_16", "DKI Jakarta")]  # cabai rawit merah, DKI Jakarta
dki.loc["2026-11-01":"2026-11-19"]

# %% [markdown]
# Lihat: 7, 8, 14, 15 November kosong (`NaN`). Itu Sabtu dan Minggu. PIHPS hanya
# mencatat hari kerja.

# %%
dki.loc["2025-11-19":].plot(figsize=(10, 3), title="Cabai rawit merah, DKI Jakarta (Rp/kg)", marker=".", lw=0.5)
plt.show()

# %% [markdown]
# ## 3. Data bolong dan forward fill
#
# Untuk menebak harga besok, kita butuh "harga terakhir yang diketahui". Di hari Sabtu,
# harga terakhirnya adalah harga Jumat. Mengisi bolong dengan nilai terakhir disebut
# **forward fill** (`ffill`).
#
# Bahayanya: kalau sebuah provinsi berhenti melapor 3 bulan, `ffill` biasa akan terus
# memakai harga 3 bulan lalu seolah-olah masih berlaku. Karena itu proyek memakai
# `ffill(limit=10)`: maksimal 10 hari, setelah itu deret dianggap **berhenti**.

# %%
contoh = pd.Series([100, np.nan, np.nan, 103] + [np.nan] * 12, index=pd.date_range("2026-01-01", periods=16))
pd.DataFrame({"asli": contoh, "ffill biasa": contoh.ffill(), "ffill(limit=10)": contoh.ffill(limit=10)})

# %% [markdown]
# Baris paling bawah: `ffill` biasa masih mengisi 103, padahal sudah 12 hari tanpa laporan.
# `ffill(limit=10)` berhenti, jadi deret yang mati tidak diberi forecast palsu.

# %% [markdown]
# ## 4. Outlier: salah ketik vs lonjakan asli
#
# Di data ada harga yang jelas salah, misalnya bawang merah di Riau tanggal 31 Maret 2017.

# %%
riau = raw[(raw["province"] == "Riau") & (raw["commodity_id"] == "com_11")].set_index("date")["price"]
riau.loc["2017-03-29":"2017-04-12"]

# %% [markdown]
# Rp 1.200 per kg bawang merah? Itu hampir pasti salah ketik (mungkin harusnya 12.000 atau 31.200).
#
# Aturan di `quality.py`: bandingkan tiap harga dengan **median 21 laporan di sekitarnya**
# (10 sebelum, dirinya, 10 sesudah). Kalau rasionya di luar **0,4×–2,5×**, tandai sebagai outlier.
#
# Kenapa median, bukan rata-rata? Satu angka ekstrem bisa menyeret rata-rata jauh,
# tapi median tidak bergeming:

# %%
angka = pd.Series([30_000, 31_000, 1_200, 30_500, 29_800])
print("rata-rata:", angka.mean(), " median:", angka.median())

# %% [markdown]
# Kenapa **di sekitarnya** (ada data sesudahnya juga), bukan cuma data sebelumnya?
# Karena kita ingin membedakan dua hal:
#
# - **Salah ketik** muncul sekali lalu hilang. Median sekitarnya tetap normal, jadi rasionya ekstrem.
# - **Lonjakan asli** bertahan berhari-hari. Median sekitarnya ikut naik, jadi rasionya biasa saja dan **lolos**.
#
# Ini penting sekali: lonjakan asli adalah hal yang ingin kita prediksi. Kalau ikut dibuang,
# model tidak pernah belajar tentang lonjakan.
#
# Buktinya, cabai rawit hijau di Sumatera Utara bulan Desember 2025 melonjak sekitar 3× dan bertahan:

# %%
sumut = raw[(raw["province"] == "Sumatera Utara") & (raw["commodity_id"] == "com_15")].set_index("date")["price"]
sumut.loc["2025-10-01":"2026-02-28"].plot(figsize=(10, 3), title="Cabai rawit hijau, Sumatera Utara")
plt.show()
flagged = quality.issues(raw)
print("Baris Sumut Des 2025 yang ditandai outlier:",
      len(flagged[(flagged["province"] == "Sumatera Utara") & (flagged["commodity_id"] == "com_15")
                  & (flagged["date"].between("2025-11-01", "2026-01-31"))]))

# %%
print(f"Total yang dibuang clean(): {len(flagged)} dari {len(raw):,} baris")
flagged[["date", "province", "commodity", "price", "rule", "ratio"]].head(10)

# %% [markdown]
# ## 5. Idempotent: aman dijalankan berkali-kali
#
# Cron harian menarik ulang 14 hari terakhir, setiap hari. Artinya, data tanggal 10 akan
# ditarik 14 kali. Kalau setiap kali ditambahkan, datanya akan dobel 14×.
#
# Solusinya **upsert**: gabungkan data lama + baru, lalu buang duplikat berdasarkan
# **key** `(date, province, commodity_id)`, pilih yang **terbaru**.

# %%
lama = pd.DataFrame({"date": ["2026-11-18", "2026-11-19"], "province": "DKI Jakarta", "commodity_id": "com_16", "price": [76_000, 76_650]})
baru = pd.DataFrame({"date": ["2026-11-19", "2026-11-20"], "province": "DKI Jakarta", "commodity_id": "com_16", "price": [77_000, 78_000]})
gabung = pd.concat([lama, baru]).drop_duplicates(["date", "province", "commodity_id"], keep="last")
gabung

# %% [markdown]
# Tanggal 19 cuma muncul sekali, dengan harga revisi (77.000). Jalankan baris di atas berapa kali pun,
# hasilnya tetap sama. Itulah **idempotent**, dan itu yang dilakukan `fetch.save()`.

# %% [markdown]
# ## Latihan
#
# 1. Berapa persen baris data mentah yang jatuh di hari Sabtu atau Minggu? (Petunjuk: `raw["date"].dt.day_of_week >= 5`)
# 2. Provinsi mana yang paling sedikit punya laporan cabai merah besar (`com_13`)?
# 3. Ubah contoh di bagian 3 supaya bolongnya 8 hari. Apa bedanya `ffill(limit=10)` sekarang?
# 4. Dengan kata-katamu sendiri: kenapa aturan outlier pakai median *di sekitar*, bukan median 21 hari *sebelumnya*?

# %%
# Kerjakan di sini


# %% [markdown]
# ## Jawaban
#
# Jalankan cell di bawah untuk 1 dan 2.
#
# 3. Dengan bolong 8 hari, `ffill(limit=10)` mengisi semuanya, karena masih di bawah batas 10 hari.
# 4. Lonjakan asli bertahan, jadi median *di sekitar* (yang ikut melihat hari-hari sesudahnya) ikut naik
#    dan lonjakan itu tidak dianggap outlier. Kalau cuma melihat ke belakang, hari pertama lonjakan
#    asli akan terlihat sama anehnya dengan salah ketik, dan ikut terbuang.

# %%
print("1.", f"{(raw['date'].dt.day_of_week >= 5).mean():.1%}", "baris di akhir pekan")
clean = quality.clean(raw)
print("2.", clean[clean["commodity_id"] == "com_13"].groupby("province").size().sort_values().head(3).to_dict())
