# %% [markdown]
# # Modul 3: Fitur, dan bahaya leakage
#
# Model cuma sepintar informasi yang kita berikan. Informasi itu disebut **fitur**.
# Di modul ini:
#
# 1. Kenapa proyek memakai **log return**, bukan selisih rupiah
# 2. Fitur **momentum** dan **volatilitas**
# 3. Fitur **mean reversion**: harga yang sudah terlalu tinggi cenderung turun
# 4. Fitur **Lebaran** dan **provinsi vs nasional**
# 5. **Leakage** di fitur, dan tes yang menjaganya
#
# Kode proyek yang dibahas: `model.panels()`, `lebaran.py`, `tests/test_spike.py`.

# %%
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
import lebaran
import model
import quality

wide = backtest.to_wide(quality.clean(quality.load_raw()))
known = wide.ffill(limit=quality.STALE_DAYS)  # harga terakhir yang diketahui, tiap hari kalender

# %% [markdown]
# ## 1. Log return: ukuran perubahan yang adil
#
# Cabai naik dari Rp 60.000 ke Rp 66.000 (+Rp 6.000). Beras naik dari Rp 13.000 ke Rp 14.300 (+Rp 1.300).
# Dalam rupiah, cabai "naik lebih banyak". Padahal **dua-duanya naik 10%**.
#
# Persen biasa juga punya masalah: naik 50% lalu turun 50% **tidak** kembali ke harga awal.

# %%
harga = 100
naik = harga * 1.5
turun = naik * 0.5
print(f"100 naik 50% = {naik}, lalu turun 50% = {turun}  (bukan 100)")
print(f"Dalam log: naik {np.log(naik / harga):+.3f}, turun {np.log(turun / naik):+.3f}, total {np.log(turun / harga):+.3f}")
print(f"Naik ke 150 lalu kembali ke 100: {np.log(150 / 100):+.3f} lalu {np.log(100 / 150):+.3f}, total {np.log(100 / 100):+.3f}")

# %% [markdown]
# Log return punya sifat bagus: **naik lalu kembali ke harga awal = total nol**, dan simetris.
# Untuk perubahan kecil, log return ≈ persen biasa (log(1,05) ≈ 0,049).
#
# Karena itu model proyek menebak `log(harga nanti / harga terakhir)`, bukan harga rupiah.
# Satu model bisa belajar dari beras dan cabai sekaligus, karena skalanya sama.

# %% [markdown]
# ## 2. Momentum dan volatilitas
#
# `model.panels()` membuat banyak fitur sekaligus. Kita lihat beberapa untuk cabai rawit merah DKI.

# %%
p = model.panels(np.log(known))
col = ("com_16", "DKI Jakarta")
fitur = pd.DataFrame({name: p[name][col] for name in ["ret_7", "ret_28", "vol_28", "dev_28", "vs_national"]})
fitur.loc["2026-11-10":"2026-11-19"].round(3)

# %% [markdown]
# - `ret_7` = log return 7 hari terakhir. Positif = sedang naik (**momentum**).
# - `vol_28` = simpangan baku perubahan harian 28 hari terakhir = **volatilitas** (seberapa liar).
# - `dev_28` = harga hari ini vs median 28 hari terakhir (log). Positif = sedang di atas normal.
# - `vs_national` = harga provinsi ini vs harga nasional komoditas yang sama.
#
# Volatilitas tiap komoditas sangat beda. Ini salah satu temuan penting proyek:

# %%
nat = np.log(known.xs(model.NATIONAL, axis=1, level="province"))
names = quality.load_raw().drop_duplicates("commodity_id").set_index("commodity_id")["commodity"]
(nat.diff().std() * np.sqrt(250)).rename(names).sort_values().round(2).plot(kind="barh", figsize=(7, 3), title="Volatilitas tahunan harga nasional")
plt.show()

# %% [markdown]
# ## 3. Momentum atau mean reversion? Tanya datanya
#
# Dua teori yang bertolak belakang:
# - **Momentum**: harga yang sedang di atas normal akan terus naik sebentar.
# - **Mean reversion**: harga yang sudah jauh di atas normal akan turun kembali.
#
# Mana yang benar untuk cabai? Jangan menebak, cek datanya. Kita kelompokkan semua hari dari 4 jenis cabai
# menurut `dev_28` (seberapa jauh harga dari median 28 harinya), lalu lihat rata-rata perubahan harga
# **14 dan 28 hari kemudian**.

# %%
future_14 = np.log(known).shift(-14) - np.log(known)  # perubahan ke depan: hanya untuk analisis!
future_28 = np.log(known).shift(-28) - np.log(known)
cabai = [c for c in wide.columns if c[0] in ("com_13", "com_14", "com_15", "com_16")]
df = pd.DataFrame({
    "dev_28": model.stack(p["dev_28"][cabai]),
    "14 hari": model.stack(future_14[cabai]), "28 hari": model.stack(future_28[cabai]),
}).dropna()
labels = ["5% terbawah", "5-20%", "20-40%", "40-60%", "60-80%", "80-95%", "5% teratas"]
df["posisi"] = pd.qcut(df["dev_28"], [0, .05, .2, .4, .6, .8, .95, 1], labels=labels)
tabel = df.groupby("posisi", observed=True)[["dev_28", "14 hari", "28 hari"]].mean().mul(100).round(1)
tabel.rename(columns={"dev_28": "posisi vs median (%)"})

# %%
tabel[["14 hari", "28 hari"]].plot(kind="bar", figsize=(8, 3), title="Perubahan harga cabai ke depan (%), menurut posisi vs median 28 hari")
plt.axhline(0, color="black", lw=0.5)
plt.show()

# %% [markdown]
# Hasilnya **dua-duanya benar, di tempat yang berbeda**:
#
# - Di **tengah** (harga sedikit di atas median), harga cenderung **terus naik** dalam 14 hari: momentum.
# - Di **5% teratas** (harga ~40% di atas median), harga cenderung **turun**, terutama dalam 28 hari: mean reversion.
#
# Hubungannya **tidak lurus**. Korelasi biasa (satu angka) akan terlihat kecil dan menyesatkan,
# padahal polanya jelas. Model berbasis **decision tree** (modul 4) bisa menangkap pola berbelok
# seperti ini, karena tree bertanya "apakah dev_28 > 0,3?" dan memberi jawaban berbeda untuk tiap kelompok.
#
# Catatan: `future_14` dan `future_28` memakai `shift(-14)` dan `shift(-28)`, yaitu data masa depan.
# Itu boleh **hanya untuk analisis** seperti ini. Sebagai fitur model, itu leakage (lihat bagian 5).

# %% [markdown]
# ## 4. Lebaran dan provinsi
#
# Model tidak tahu kalender. Kita beri tahu lewat dua angka: **berapa hari lagi** Lebaran dan
# **sudah berapa hari sejak** Lebaran terakhir, dibatasi maksimal 60 (lebih jauh dari itu tidak berpengaruh).

# %%
tanggal = pd.DatetimeIndex(["2026-02-01", "2026-03-14", "2026-03-20", "2026-03-21", "2026-03-25", "2026-07-01"])
until, since = lebaran.distances(tanggal)
pd.DataFrame({"hari menuju": until, "hari sejak": since}, index=tanggal.date)

# %% [markdown]
# Fitur kategori **commodity_id** dan **province** juga diberikan. Dengan itu, satu model global tahu
# bahwa beras jarang bergerak, cabai liar, dan Maluku Utara selalu lebih mahal dari NTB.

# %% [markdown]
# ## 5. Leakage di fitur
#
# Fitur di tanggal X **hanya boleh** memakai data sampai tanggal X. Kesalahan yang sangat umum:
# rolling window yang **centered** (melihat ke depan juga).

# %%
s = np.log(known[col])
salah = s - s.rolling(28, center=True).median()  # melihat 14 hari ke depan
benar = s - s.rolling(28).median()  # hanya ke belakang
target = s.shift(-7) - s  # perubahan 7 hari ke depan
print("Korelasi fitur dengan masa depan:")
print(f"  centered (bocor): {salah.corr(target):+.2f}")
print(f"  trailing (jujur): {benar.corr(target):+.2f}")

# %% [markdown]
# Fitur yang bocor tampak jauh lebih "berguna", karena sebagian isinya adalah jawaban.
# Di backtest model akan tampak hebat, lalu gagal di produksi, karena di produksi data masa depan tidak ada.
#
# **Cara proyek menjaga ini:** `tests/test_spike.py` menghitung fitur dua kali, sekali dengan data penuh dan
# sekali dengan data yang dipotong di tanggal X, lalu memastikan nilai di tanggal X **persis sama**.
# Kalau suatu hari ada yang mengubah fitur jadi bocor, tes itu gagal.

# %%
cut = known.index[-200]
full_p = model.panels(np.log(known))
past_p = model.panels(np.log(known.loc[:cut]))
print("Semua fitur sama di tanggal potong:", all(full_p[k].loc[cut].equals(past_p[k].loc[cut]) for k in full_p))
s_full, s_cut = known[col], known[col].loc[:cut]
print("Versi centered sama?", (s_full.rolling(28, center=True).median().loc[cut], s_cut.rolling(28, center=True).median().loc[cut]))

# %% [markdown]
# ## Latihan
#
# 1. Hitung log return dari Rp 40.000 ke Rp 60.000, lalu dari Rp 60.000 kembali ke Rp 40.000. Jumlahkan.
# 2. Ulangi analisis bagian 3 untuk **beras** (`com_3`), cukup 5 kelompok. Momentum, mean reversion, atau tidak ada pola?
# 3. Kenapa fitur `lebaran_until` dibatasi 60 hari, bukan dibiarkan sampai 300+?
# 4. Kalau rekan kerja bilang "skornya naik dari 0,5 jadi 0,9 setelah saya tambah fitur rata-rata 7 hari",
#    apa yang pertama kali kamu cek?

# %%
# Kerjakan di sini


# %% [markdown]
# ## Jawaban
#
# 1. log(60/40) = +0,405, log(40/60) = −0,405, jumlah 0. (Dalam persen biasa: +50% lalu −33%.)
# 2. Jalankan cell di bawah: perubahannya sangat kecil (di bawah 0,5% dalam 14 hari), dan yang di atas median
#    sedikit cenderung terus naik, tidak ada mean reversion. Harga beras bergerak pelan dan cenderung searah.
# 3. Lebaran 10 bulan lagi tidak memengaruhi harga hari ini. Tanpa batas, model bisa menemukan pola kebetulan
#    di angka 200 atau 300 yang tidak bermakna.
# 4. Apakah rata-rata 7 harinya **trailing** (ke belakang) atau **centered/ke depan**. Lompatan skor sebesar itu
#    hampir selalu tanda leakage.

# %%
beras = [c for c in wide.columns if c[0] == "com_3"]
b = pd.DataFrame({"dev_28": model.stack(p["dev_28"][beras]), "ke_depan_14": model.stack(future_14[beras])}).dropna()
b["kelompok"] = pd.qcut(b["dev_28"].rank(method="first"), 5, labels=["jauh di bawah", "di bawah", "normal", "di atas", "jauh di atas"])  # rank: banyak nilai 0 kembar
print(b.groupby("kelompok", observed=True)["ke_depan_14"].mean().mul(100).round(2))
