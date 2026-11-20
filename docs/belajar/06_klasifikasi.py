# %% [markdown]
# # Modul 6: Peringatan dini lonjakan (klasifikasi)
#
# Forecast menjawab "berapa harganya nanti". Peringatan dini menjawab pertanyaan yang lebih dekat ke keputusan:
# **"apakah deret ini akan melonjak dalam 14 hari? Ya atau tidak?"** Ini **klasifikasi**.
#
# 1. Mendefinisikan "lonjakan" dengan ambang per komoditas
# 2. Label: lonjakan **baru** dalam 14 hari
# 3. Kelas tidak seimbang, dan kenapa akurasi menipu
# 4. Precision, recall, dan cut-off, dihitung dengan tangan
# 5. Average precision: classifier vs heuristik vs acak
#
# Kode proyek yang dibahas: `spike.py`.

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
import spike
from model import stack

wide = backtest.to_wide(quality.clean(quality.load_raw()))
names = quality.load_raw().drop_duplicates("commodity_id").set_index("commodity_id")["commodity"]

# %% [markdown]
# ## 1. Apa itu "lonjakan"?
#
# **Batas waspada** = median 20 laporan terakhir (~4 minggu) × (1 + ambang). Ambangnya beda per komoditas:

# %%
pd.Series(spike.THRESHOLDS).rename(index=names).map("+{:.0%}".format).to_frame("ambang")

# %% [markdown]
# Kenapa tidak satu ambang (misal 10%) untuk semua? Karena volatilitasnya beda jauh (modul 3).
# Dengan 10%, cabai rawit merah "waspada" di 28% hari (alarm yang bunyi terus akhirnya diabaikan),
# sedangkan beras hampir tidak pernah.
#
# Ambang di atas dipilih dari data 2017–2024 sehingga deret yang tenang berubah jadi lonjakan dalam 14 hari
# di **sekitar 10%** hari, untuk tiap komoditas. Satu aturan yang adil, bukan satu angka.

# %%
level = spike.levels(wide)
col = ("com_11", "Bali")  # bawang merah Bali
ax = wide[col].loc["2025-06-01":].plot(figsize=(10, 3), marker=".", lw=0.5, label="harga")
level[col].loc["2025-06-01":].plot(ax=ax, color="red", ls="--", label="batas waspada")
ax.set_title("Bawang merah, Bali: harga vs batas waspada (median 20 laporan × 1,20)")
ax.legend()
plt.show()

# %% [markdown]
# ## 2. Label: lonjakan BARU dalam 14 hari
#
# Untuk setiap hari dan deret:
# - Kalau harga **sudah** di atas batas, deret itu sedang *waspada*. Itu sudah kelihatan, tidak perlu diprediksi.
# - Kalau masih **tenang**: label = 1 kalau ada laporan dalam 14 hari ke depan yang menembus batas, 0 kalau tidak.
#
# Label hari ini baru diketahui **14 hari lagi**. Hari-hari terakhir di data belum punya label (`NaN`).

# %%
calm, label, ratio = spike.labels(wide)
pd.DataFrame({"harga": wide[col], "batas": level[col].round(), "tenang?": calm[col], "label": label[col]}).loc["2026-10-27":].dropna(subset=["harga"])

# %% [markdown]
# ## 3. Kelas tidak seimbang, dan akurasi yang menipu

# %%
recent = label.loc["2024-11":]
y = stack(recent[recent.index.day_of_week < 5]).dropna()  # hari kerja saja, seperti backtest proyek
print(f"{len(y):,} hari-deret tenang, {y.mean():.1%} berubah jadi lonjakan dalam 14 hari")
print(f"Model yang SELALU bilang 'tidak akan melonjak': akurasi {1 - y.mean():.1%}")

# %% [markdown]
# Akurasi 92% tanpa pernah memperingatkan apa pun. Model itu tidak berguna, tapi akurasinya tinggi.
# Kalau kejadiannya jarang (**kelas tidak seimbang**), akurasi menipu. Kita butuh ukuran lain.

# %% [markdown]
# ## 4. Precision, recall, cut-off
#
# Kita punya skor untuk tiap deret (makin tinggi, makin mungkin melonjak). Kita nyalakan peringatan kalau
# skor ≥ **cut-off**. Lalu ada 4 kemungkinan:
#
# | | Benar melonjak | Tidak melonjak |
# |---|---|---|
# | **Diperingatkan** | TP (tepat) | FP (alarm palsu) |
# | **Tidak diperingatkan** | FN (kecolongan) | TN |
#
# - **Precision** = TP / (TP + FP): dari semua peringatan, berapa yang benar?
# - **Recall** = TP / (TP + FN): dari semua lonjakan, berapa yang sempat diperingatkan?
#
# Pakai skor **heuristik** dulu (tanpa machine learning): `ratio` = harga hari ini / batas waspada.
# Intuisinya: makin dekat ke batas, makin mungkin menembusnya.

# %%
score = stack(ratio.loc["2024-11":]).reindex(y.index)
rows = []
for cut in (0.80, 0.85, 0.90, 0.95):
    warn = score >= cut
    tp = int((warn & (y == 1)).sum())
    fp = int((warn & (y == 0)).sum())
    fn = int((~warn & (y == 1)).sum())
    rows.append({"cut-off (harga/batas)": cut, "peringatan": tp + fp, "TP": tp, "FP": fp, "FN": fn,
                 "precision": tp / max(tp + fp, 1), "recall": tp / (tp + fn)})
pd.DataFrame(rows).round(3)

# %% [markdown]
# Lihat **tarik-menariknya**: cut-off rendah = banyak peringatan, recall tinggi tapi precision rendah
# (banyak alarm palsu). Cut-off tinggi = sedikit peringatan, lebih tepat, tapi banyak yang kecolongan.
#
# Tidak ada cut-off yang "benar" secara teknis. Itu **keputusan bisnis**: user lebih suka daftar pendek yang
# hampir pasti benar, atau daftar panjang yang menangkap lebih banyak?

# %% [markdown]
# ## 5. Average precision: satu angka untuk semua cut-off
#
# Urutkan semua hari-deret dari skor tertinggi. Berjalan dari atas ke bawah, setiap kali ketemu lonjakan asli,
# catat precision saat itu. **Average precision (AP)** = rata-rata catatan itu.
#
# - Skor acak: AP ≈ base rate (7–8%).
# - Skor sempurna (semua lonjakan di urutan teratas): AP = 1.

# %%
print(f"AP acak       ≈ {y.mean():.3f}")
print(f"AP heuristik  = {spike.average_precision(score.to_numpy(), y.to_numpy()):.3f}")

# %% [markdown]
# Sekarang classifier LightGBM dari `spike.py`. Fiturnya sama seperti forecaster ditambah `to_level`
# (= heuristik di atas), dan dilatih ulang tiap 6 bulan hanya dengan data yang labelnya sudah diketahui.
# Backtest-nya butuh ~40 detik.

# %%
bt = spike.run(wide)
print(f"{bt['date'].nunique()} hari, {len(bt):,} hari-deret tenang, lonjakan {bt['spike'].mean():.1%}")
print(f"AP acak       ≈ {bt['spike'].mean():.3f}")
print(f"AP heuristik  = {spike.average_precision(bt['to_level'].to_numpy(), bt['spike'].to_numpy()):.3f}")
print(f"AP classifier = {spike.average_precision(bt['p'].to_numpy(), bt['spike'].to_numpy()):.3f}")

# %%
rows = []
for cut in (0.3, 0.4, 0.5, 0.6, 0.7):
    a = bt[bt["p"] >= cut]
    rows.append({"cut-off peluang": cut, "peringatan/hari": len(a) / bt["date"].nunique(),
                 "precision": a["spike"].mean(), "recall": a["spike"].sum() / bt["spike"].sum()})
pd.DataFrame(rows).round(3)

# %% [markdown]
# Classifier jauh lebih baik dari heuristik dan acak. Dashboard memakai cut-off **0,5**: ~9 peringatan per hari,
# ~65% benar-benar melonjak, menangkap ~27% lonjakan.
#
# Apakah 65% bagus? Bandingkan dengan base rate 7,5%: tanpa model, memilih 9 deret acak hanya benar ~1 dari 13.
# Dengan model, ~6 dari 9. Dan cut-off bisa digeser sesuai kebutuhan user.

# %% [markdown]
# ## Latihan
#
# 1. Dari tabel cut-off classifier, pilih cut-off untuk user yang cuma punya waktu mengecek **3 deret per hari**.
#    Berapa precision-nya?
# 2. Hitung AP heuristik khusus untuk cabai rawit merah (`com_16`) dari `bt`. Lebih tinggi atau lebih rendah dari keseluruhan?
# 3. Kenapa deret yang **sudah** di atas batas tidak dimasukkan ke data latih classifier?
# 4. Jelaskan ke orang awam, dalam 2 kalimat, beda precision dan recall memakai contoh alarm kebakaran.

# %%
# Kerjakan di sini


# %% [markdown]
# ## Jawaban
#
# 1. Sekitar cut-off 0,7 (~2,7 peringatan/hari, precision ~81%).
# 2. Jalankan cell di bawah.
# 3. Karena lonjakan itu sudah terjadi dan sudah kelihatan di tabel waspada. Memasukkannya membuat tugas
#    terlihat mudah ("yang sudah tinggi akan tetap tinggi") dan skor jadi bagus palsu, padahal nilai peringatan
#    dini ada di deret yang belum kelihatan bermasalah.
# 4. Precision: dari semua alarm yang berbunyi, berapa yang memang ada kebakaran. Recall: dari semua kebakaran,
#    berapa yang alarmnya sempat berbunyi.

# %%
g = bt[bt["commodity_id"] == "com_16"]
print("2.", f"heuristik {spike.average_precision(g['to_level'].to_numpy(), g['spike'].to_numpy()):.3f},",
      f"classifier {spike.average_precision(g['p'].to_numpy(), g['spike'].to_numpy()):.3f}")
