# %% [markdown]
# # Modul 2: Forecast, baseline, dan backtest
#
# Pertanyaan modul ini: **bagaimana kita tahu sebuah cara menebak itu bagus?**
#
# 1. Apa itu origin, horizon, dan target
# 2. Baseline: cara menebak paling bodoh yang harus dikalahkan
# 3. MAPE: mengukur seberapa meleset
# 4. Kenapa data deret waktu **tidak boleh** diacak (leakage)
# 5. Backtest rolling-origin: memutar waktu mundur berkali-kali
#
# Kode proyek yang dibahas: `backtest.py`.

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
import quality

wide = backtest.to_wide(quality.clean(quality.load_raw()))
dki = wide[("com_16", "DKI Jakarta")]  # cabai rawit merah DKI

# %% [markdown]
# ## 1. Origin, horizon, target
#
# Bayangkan hari ini **Senin, 3 November 2025**. Kita cuma tahu harga sampai hari ini.
# Kita ingin menebak harga 14 hari ke depan.
#
# - **Origin** = 3 Nov 2025 (hari tebakan dibuat)
# - **Horizon** = 1, 2, ..., 14 hari setelah origin
# - **Target** = harga asli di hari-hari itu (yang baru kita ketahui belakangan)

# %%
origin = pd.Timestamp("2025-11-03")
history = dki.loc[:origin]  # hanya ini yang boleh dilihat
future = dki.loc[origin + pd.Timedelta(days=1):origin + pd.Timedelta(days=14)].dropna()  # jawaban, disembunyikan
print("Harga terakhir yang diketahui:", history.dropna().iloc[-1])
future

# %% [markdown]
# ## 2. Baseline naive: "besok sama dengan hari ini"
#
# Tebakan paling sederhana: semua hari ke depan = harga terakhir.
# Kedengarannya bodoh, tapi harga pangan biasanya bergerak pelan, jadi tebakan ini sering cukup dekat.
# Karena itu baseline ini jadi **standar minimum**: model yang tidak bisa mengalahkannya tidak berguna.

# %%
last = history.dropna().iloc[-1]
tebakan = pd.Series(last, index=future.index)
pd.DataFrame({"asli": future, "tebakan naive": tebakan, "meleset (Rp)": future - tebakan})

# %% [markdown]
# ## 3. MAPE: meleset berapa persen?
#
# Meleset Rp 5.000 untuk cabai (Rp 60 ribuan) itu kecil, tapi untuk beras (Rp 13 ribuan) besar sekali.
# Supaya bisa dibandingkan antar komoditas, error dihitung dalam **persen dari harga asli**:
#
# `APE = |tebakan - asli| / asli`, lalu dirata-rata = **MAPE** (*mean absolute percentage error*).

# %%
ape = (tebakan - future).abs() / future
print(f"MAPE untuk origin ini: {ape.mean():.1%}")
print("Fungsi yang sama di proyek:", round(backtest.mape(tebakan, future), 1), "%")

# %% [markdown]
# ## 4. Kenapa tidak boleh diacak
#
# Di machine learning biasa, data sering diacak lalu dibagi 80% latih, 20% uji.
# Untuk deret waktu itu **curang**. Contoh: kita acak hari-hari tahun 2025, lalu "model" menebak
# harga hari uji dengan rata-rata hari latih tepat sebelum dan sesudahnya.

# %%
s = dki.loc["2025-01-01":"2025-12-31"].dropna()
rng = np.random.default_rng(0)
is_test = rng.random(len(s)) < 0.2
train = s[~is_test]

# tebakan "curang": rata-rata tetangga latih terdekat sebelum dan SESUDAH
curang = train.reindex(s.index).interpolate()[is_test]
# tebakan jujur: harga latih terakhir SEBELUM hari itu
jujur = train.reindex(s.index).ffill()[is_test]
actual = s[is_test]
print(f"MAPE 'acak + lihat tetangga': {backtest.mape(curang, actual):.2f}%")
print(f"MAPE naive jujur:            {backtest.mape(jujur, actual):.2f}%")

# %% [markdown]
# Kelihatannya model "acak" lebih bagus. Tapi di dunia nyata, **saat kita menebak hari Selasa,
# harga hari Rabu belum ada**. Skor bagus itu palsu, karena model melihat masa depan.
# Ini disebut **leakage** (kebocoran). Hasilnya: model terlihat hebat di laporan, lalu
# hancur saat dipakai sungguhan.
#
# Aturan untuk deret waktu: **yang diuji harus selalu setelah yang dipakai belajar.**

# %% [markdown]
# ## 5. Backtest rolling-origin
#
# Satu origin bisa kebetulan beruntung atau sial. Jadi kita ulangi berkali-kali:
# pura-pura hari ini Senin minggu ke-1, tebak 14 hari, nilai; geser ke Senin berikutnya; ulangi.
#
# Di proyek ini: **104 Senin** selama 2 tahun, untuk semua deret sekaligus.
# Di sini kita coba versi kecil: satu deret, satu tahun.

# %%
hasil = []
for o in pd.date_range("2025-01-06", "2025-12-15", freq="W-MON"):
    hist = dki.loc[:o].ffill(limit=10)
    if pd.isna(hist.iloc[-1]):
        continue
    akan_datang = dki.loc[o + pd.Timedelta(days=1):o + pd.Timedelta(days=14)].dropna()
    for d, harga in akan_datang.items():
        hasil.append({"origin": o, "h": (d - o).days, "asli": harga, "naive": hist.iloc[-1]})
bt = pd.DataFrame(hasil)
bt["ape"] = (bt["naive"] - bt["asli"]).abs() / bt["asli"] * 100
print(f"{bt['origin'].nunique()} origin, {len(bt)} tebakan dinilai")
bt.groupby("h")["ape"].mean().plot(kind="bar", figsize=(8, 3), title="MAPE naive per horizon, cabai rawit merah DKI 2025 (%)")
plt.show()

# %% [markdown]
# Pola paling penting: **makin jauh horizon, makin besar error**. Wajar: menebak besok gampang,
# menebak 2 minggu lagi susah. Karena itu proyek selalu melaporkan error per kelompok horizon
# (1–3, 4–7, 8–14 hari), bukan satu angka rata-rata.

# %% [markdown]
# ## 6. Baseline kedua: meniru tahun lalu (seasonal)
#
# Ide: "Tahun lalu di periode ini cabai naik 10%, jadi tahun ini juga naik 10%."
# Kedengarannya pintar. Ayo uji di backtest proyek (semua deret, 2 tahun, ~45 detik).

# %%
full = backtest.run(quality.clean(quality.load_raw()), [backtest.naive, backtest.seasonal])
full = full.assign(horizon=pd.cut(full["h"], [0, 3, 7, 14], labels=["1-3", "4-7", "8-14"]))
full.groupby("horizon", observed=True).apply(
    lambda g: pd.Series({"naive": backtest.mape(g["naive"], g["actual"]), "seasonal": backtest.mape(g["seasonal"], g["actual"])})
).round(1)

# %% [markdown]
# Seasonal **kalah di semua horizon**. Pergerakan harga tahun lalu tidak berulang di tanggal yang sama tahun ini
# (hujan, panen, dan Lebaran bergeser tiap tahun). Ide yang kedengarannya masuk akal belum tentu benar,
# dan cara membuktikannya adalah backtest.
#
# Pelajaran besarnya: **"harga terakhir" adalah baseline yang kuat.** Semua model di modul berikutnya
# harus mengalahkan angka naive di tabel ini.

# %% [markdown]
# ## Latihan
#
# 1. Ulangi bagian 5 untuk **beras** di DKI (`com_3`). MAPE-nya jauh lebih kecil atau lebih besar? Kenapa?
# 2. Di bagian 5, hitung MAPE rata-rata hanya untuk h ≤ 3 dan h ≥ 8. Berapa kali lipat bedanya?
# 3. Dengan kata-katamu sendiri: kenapa skor "acak + lihat tetangga" di bagian 4 tidak boleh dipercaya?
# 4. Kalau ada model dengan MAPE 7% di horizon 8–14 hari, apakah itu bagus? Bandingkan dengan tabel bagian 6.

# %%
# Kerjakan di sini


# %% [markdown]
# ## Jawaban
#
# 1. Jalankan cell di bawah. Beras jauh lebih stabil (MAPE ~0,3%), jadi hampir tidak ada yang bisa dimenangkan model.
# 2. Lihat cell di bawah. Error 8–14 hari beberapa kali lipat error 1–3 hari.
# 3. Karena ia memakai harga sesudah hari yang ditebak, yang di dunia nyata belum ada.
# 4. Tidak. Naive sudah ~6,3% di horizon itu, jadi model 7% justru lebih buruk dari "harga terakhir".

# %%
beras = wide[("com_3", "DKI Jakarta")]
ape_beras = []
for o in pd.date_range("2025-01-06", "2025-12-15", freq="W-MON"):
    last_b = beras.loc[:o].ffill(limit=10).iloc[-1]
    nxt = beras.loc[o + pd.Timedelta(days=1):o + pd.Timedelta(days=14)].dropna()
    ape_beras += list(((last_b - nxt).abs() / nxt * 100).values)
print(f"1. MAPE naive beras DKI 2025: {np.mean(ape_beras):.2f}%  (cabai: {bt['ape'].mean():.2f}%)")
print(f"2. h<=3: {bt[bt.h <= 3]['ape'].mean():.1f}%   h>=8: {bt[bt.h >= 8]['ape'].mean():.1f}%")
