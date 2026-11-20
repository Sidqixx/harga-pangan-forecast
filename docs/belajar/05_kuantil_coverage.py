# %% [markdown]
# # Modul 5: Kuantil, P10/P50/P90, dan coverage
#
# Satu angka tebakan menyembunyikan hal penting: **seberapa yakin** kita. "Cabai minggu depan Rp 77 ribu"
# bisa berarti "pasti antara 76–78 ribu" atau "bisa saja 60 sampai 100 ribu". Untuk keputusan, bedanya besar.
#
# 1. Apa itu kuantil
# 2. Rentang P10–P90 dari sejarah
# 3. **Pinball loss**: cara melatih model supaya menebak kuantil
# 4. **Coverage**: mengecek apakah rentangnya jujur
#
# Kode proyek yang dibahas: `model.py` (`QUANTILES`, `objective="quantile"`), `backtest.summary()`.

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
import model
import quality

wide = backtest.to_wide(quality.clean(quality.load_raw()))
known = wide.ffill(limit=quality.STALE_DAYS)

# %% [markdown]
# ## 1. Kuantil
#
# Urutkan semua angka dari kecil ke besar. **Kuantil 10%** (P10) adalah angka yang 10% data ada di bawahnya.
# **P50** adalah median. **P90**: 90% data di bawahnya.

# %%
nilai = pd.Series([3, 7, 1, 9, 4, 6, 2, 8, 10, 5])
print("diurutkan:", sorted(nilai))
print("P10 =", nilai.quantile(0.1), " P50 =", nilai.quantile(0.5), " P90 =", nilai.quantile(0.9))

# %% [markdown]
# ## 2. Rentang dari sejarah: cabai vs beras
#
# Lihat semua perubahan harga **7 hari** yang pernah terjadi, untuk cabai rawit merah dan beras di semua provinsi.

# %%
change_7 = (known.shift(-7) / known - 1).loc["2018":"2024"]  # hanya untuk analisis sejarah
cabai = change_7[[c for c in wide.columns if c[0] == "com_16"]].stack(["commodity_id", "province"], future_stack=True).dropna()
beras = change_7[[c for c in wide.columns if c[0] == "com_3"]].stack(["commodity_id", "province"], future_stack=True).dropna()
pd.DataFrame({
    "cabai rawit merah": cabai.quantile([0.1, 0.5, 0.9]),
    "beras": beras.quantile([0.1, 0.5, 0.9]),
}, ).rename(index={0.1: "P10", 0.5: "P50", 0.9: "P90"}).map(lambda v: f"{v:+.1%}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
cabai.clip(-0.5, 0.8).hist(bins=80, ax=ax[0])
ax[0].set_title("Cabai rawit merah: perubahan 7 hari")
beras.clip(-0.05, 0.05).hist(bins=80, ax=ax[1])
ax[1].set_title("Beras: perubahan 7 hari")
for a, s in zip(ax, (cabai, beras)):
    for q in (0.1, 0.9):
        a.axvline(s.quantile(q), color="orange", ls="--")
plt.show()

# %% [markdown]
# Dua hal penting:
# - Rentang 80% cabai **jauh lebih lebar** dari beras. Satu rentang untuk semua komoditas jelas salah.
# - Rentang cabai **miring ke atas**: P90 lebih jauh dari nol dibanding P10. Cabai lebih sering melonjak tajam
#   daripada anjlok tajam. Itu sebabnya P90 di forecast proyek sering jauh di atas P50.
#
# Tapi rentang dari sejarah ini **sama untuk setiap hari**. Padahal menjelang Lebaran, atau saat harga sudah
# melonjak, ketidakpastiannya beda. Kita ingin model yang menebak kuantil **sesuai kondisi hari itu**.

# %% [markdown]
# ## 3. Pinball loss: mengajari model menebak kuantil
#
# Model biasa belajar meminimalkan kesalahan kuadrat, dan hasilnya adalah **rata-rata**.
# Untuk menebak **P90**, kita pakai hukuman yang **tidak simetris**:
#
# - tebakan **terlalu rendah** (harga asli di atas tebakan): hukuman × 0,9
# - tebakan **terlalu tinggi**: hukuman × 0,1
#
# Jadi menebak terlalu rendah dihukum **9 kali lebih berat**. Model akhirnya menebak cukup tinggi sehingga
# hanya ~10% harga asli yang melebihinya. Itulah P90. Hukuman ini disebut **pinball loss** (atau quantile loss).

# %%
def pinball(tebakan, asli, q):
    selisih = asli - tebakan
    return np.mean(np.where(selisih > 0, q * selisih, (q - 1) * selisih))


# Bukti: cari satu angka yang meminimalkan pinball loss q=0.9 untuk perubahan 7 hari cabai
kandidat = np.linspace(-0.3, 0.5, 801)
sampel = cabai.sample(20_000, random_state=0).to_numpy()
loss = [pinball(c, sampel, 0.9) for c in kandidat]
terbaik = kandidat[int(np.argmin(loss))]
print(f"Angka dengan pinball loss (q=0.9) terkecil: {terbaik:+.3f}")
print(f"P90 sungguhan dari data:                    {np.quantile(sampel, 0.9):+.3f}")
plt.figure(figsize=(7, 3))
plt.plot(kandidat, loss)
plt.axvline(terbaik, color="orange", ls="--")
plt.title("Pinball loss q=0.9 untuk tiap tebakan konstan")
plt.show()

# %% [markdown]
# Angka yang meminimalkan pinball loss = P90. Jadi kalau LightGBM dilatih dengan pinball loss q=0.9
# (`objective="quantile", alpha=0.9`), ia belajar menebak P90 **untuk setiap kondisi**: P90 saat harga tenang
# akan sempit, P90 saat harga sedang liar akan lebar.
#
# Proyek melatih **3 model**: q=0,1, q=0,5, q=0,9 → P10, P50, P90.
# Kadang tiga model itu "bertabrakan" (P10 di atas P50); `model.py` mengurutkan ulang ketiganya supaya tidak terjadi.

# %% [markdown]
# ## 4. Coverage: apakah rentangnya jujur?
#
# Model bilang "80% yakin harga ada di antara P10 dan P90". Cara mengeceknya: dari semua tebakan di backtest,
# hitung berapa persen harga asli **benar-benar** jatuh di dalam rentang. Itu **coverage**.
#
# Kita coba dulu dengan rentang paling sederhana: P10/P90 dari sejarah (bagian 2), dipasang di sekitar
# harga terakhir, dan diuji di 2025–2026 (data yang tidak dipakai untuk menghitung rentangnya).

# %%
lo, hi = cabai.quantile(0.1), cabai.quantile(0.9)
test = pd.DataFrame({
    "asli": known.shift(-7).loc["2025":][[c for c in wide.columns if c[0] == "com_16"]].stack(["commodity_id", "province"], future_stack=True),
    "terakhir": known.loc["2025":][[c for c in wide.columns if c[0] == "com_16"]].stack(["commodity_id", "province"], future_stack=True),
}).dropna()
inside = test["asli"].between(test["terakhir"] * (1 + lo), test["terakhir"] * (1 + hi))
print(f"Coverage rentang sejarah, cabai rawit merah 2025-2026: {inside.mean():.0%} (target 80%)")

# %% [markdown]
# Lumayan, mendekati 80%. Tapi itu **rata-rata**. Pecah menurut kondisi pasar: hari yang harganya sedang tenang
# vs sedang liar (pakai fitur `vol_28` dari modul 3).

# %%
vol = model.panels(np.log(known))["vol_28"].loc["2025":][[c for c in wide.columns if c[0] == "com_16"]]
test["kondisi"] = pd.qcut(vol.stack(["commodity_id", "province"], future_stack=True).reindex(test.index), 3, labels=["tenang", "sedang", "liar"])
inside.groupby(test["kondisi"], observed=True).mean().map("{:.0%}".format)

# %% [markdown]
# Rentang tetap dari sejarah **kelebaran saat tenang** (~90%) dan **kesempitan saat liar** (~75%), justru
# saat rentang itu paling dibutuhkan. Coverage rata-rata 80% menyembunyikan masalah ini.
#
# Model kuantil melihat fitur seperti volatilitas dan Lebaran, jadi rentangnya **menyesuaikan kondisi**:
# sempit saat tenang, lebar saat liar. Itu alasan memakai model untuk kuantil, bukan sekadar angka sejarah.

# %% [markdown]
# | Coverage | Artinya |
# |---|---|
# | ~80% | Jujur |
# | jauh di bawah 80% | **Terlalu pede**: rentang kesempitan, harga asli sering keluar |
# | jauh di atas 80% | **Terlalu hati-hati**: rentang kelebaran, kurang berguna |
#
# Hasil model proyek di backtest 2 tahun (dari `backtest.py`):
#
# | Horizon | Coverage P10–P90 |
# |---|---|
# | 1–3 hari | 80,5% |
# | 4–7 hari | 79,8% |
# | 8–14 hari | 78,5% |
#
# Hampir persis 80%: model **terkalibrasi**. Hanya untuk cabai di 8–14 hari coverage-nya 73–78%,
# sedikit terlalu pede: lonjakan cabai kadang lebih besar dari perkiraan model.
#
# Coverage juga dipakai di **gerbang rilis**: model baru hanya boleh dipublikasikan kalau coverage-nya
# di 4 minggu terakhir ada di antara 70% dan 90% (modul 7).

# %% [markdown]
# ## Latihan
#
# 1. Hitung P10/P50/P90 perubahan **14 hari** untuk bawang merah (`com_11`). Lebih lebar dari 7 hari?
# 2. Di bagian 3, ganti q=0,9 menjadi q=0,1. Angka terbaiknya mendekati apa?
# 3. Ubah rentang di bagian 4 menjadi P25–P75 sejarah. Berapa coverage yang *seharusnya*, dan berapa hasilnya?
# 4. Kenapa rentang yang terlalu lebar juga buruk, padahal coverage-nya tinggi?

# %%
# Kerjakan di sini


# %% [markdown]
# ## Jawaban
#
# 1. Jalankan cell di bawah. Rentang 14 hari lebih lebar, karena makin jauh ke depan makin tidak pasti.
# 2. Mendekati P10, sesuai q yang dipakai.
# 3. Seharusnya ~50% (75 − 25). Jalankan cell di bawah untuk hasilnya.
# 4. Rentang "cabai minggu depan antara Rp 20 ribu dan Rp 200 ribu" selalu benar tapi tidak membantu keputusan apa pun.
#    Rentang yang berguna itu **sesempit mungkin, tapi tetap jujur**.

# %%
c14 = (known.shift(-14) / known - 1).loc["2018":"2024"][[c for c in wide.columns if c[0] == "com_11"]]
print("1.", c14.stack(["commodity_id", "province"], future_stack=True).dropna().quantile([0.1, 0.5, 0.9]).round(3).to_dict())
lo25, hi75 = cabai.quantile(0.25), cabai.quantile(0.75)
print("3.", f"{test['asli'].between(test['terakhir'] * (1 + lo25), test['terakhir'] * (1 + hi75)).mean():.0%}")
