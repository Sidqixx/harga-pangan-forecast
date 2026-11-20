# %% [markdown]
# # Modul 4: Dari decision tree ke LightGBM, dibangun dari nol
#
# LightGBM kelihatan seperti kotak ajaib. Padahal idenya sederhana, dan di modul ini kita bangun sendiri
# versi kecilnya dengan numpy, lalu membandingkannya dengan LightGBM asli.
#
# 1. Soal latihan: tebak perubahan harga cabai 14 hari ke depan
# 2. **Split**: satu pertanyaan ya/tidak terbaik
# 3. **Boosting**: ratusan split kecil yang saling memperbaiki
# 4. LightGBM asli, dan **overfitting**
#
# Kode proyek yang dibahas: `model.py` (`LightGBM`, `PARAMS`).

# %%
import os
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "quality.py").exists())
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest
import model
import quality

# %% [markdown]
# ## 1. Soalnya
#
# Untuk semua deret cabai (4 jenis × semua provinsi), di setiap Senin, kita punya 3 fitur:
# `dev_28` (posisi vs median), `ret_7` (momentum seminggu), `vol_28` (volatilitas).
# Target: perubahan harga **14 hari kemudian**, dalam log (≈ persen).
#
# Data latih: Senin sebelum Nov 2024. Data uji: Senin sesudahnya. (Uji selalu **setelah** latih, modul 2.)

# %%
wide = backtest.to_wide(quality.clean(quality.load_raw()))
known = wide.ffill(limit=quality.STALE_DAYS)
logp = np.log(known)
p = model.panels(logp)
cabai = [c for c in wide.columns if c[0] in ("com_13", "com_14", "com_15", "com_16")]
mondays = logp.index[logp.index.day_of_week == 0]
FEATURES = ["dev_28", "ret_7", "vol_28"]

data = pd.DataFrame({f: model.stack(p[f].loc[mondays, cabai]) for f in FEATURES})
data["y"] = model.stack((logp.shift(-14) - logp).loc[mondays, cabai])
data = data.dropna().reset_index()
is_train = data["date"] < "2024-11-01"
X_train, y_train = data.loc[is_train, FEATURES].to_numpy(), data.loc[is_train, "y"].to_numpy()
X_test, y_test = data.loc[~is_train, FEATURES].to_numpy(), data.loc[~is_train, "y"].to_numpy()
print(f"latih: {len(y_train):,} baris, uji: {len(y_test):,} baris")


def mae(pred, y):
    """Rata-rata error absolut, dalam %. Untuk log return kecil, ini ≈ MAPE."""
    return float(np.mean(np.abs(pred - y)) * 100)


print(f"Tebak 0 (= naive, harga tidak berubah): error uji {mae(np.zeros_like(y_test), y_test):.2f}%")

# %% [markdown]
# "Tebak 0" sama dengan baseline naive: harga 14 hari lagi = harga sekarang. Itu angka yang harus dikalahkan.

# %% [markdown]
# ## 2. Split: pertanyaan ya/tidak terbaik
#
# Decision tree paling kecil (disebut *stump*) cuma punya satu pertanyaan, misalnya
# **"apakah dev_28 > 0,25?"**. Kalau ya, tebakannya rata-rata y untuk kelompok "ya"; kalau tidak,
# rata-rata kelompok "tidak".
#
# Pertanyaan terbaik = yang membuat dua kelompok **paling beda** satu sama lain, sehingga total
# kesalahan kuadrat (*SSE*) paling kecil. Caranya brute force: coba semua batas, pilih yang terbaik.

# %%
def best_split(X, r, min_leaf=500):
    """Cari (fitur, batas) yang paling menurunkan SSE dari target r. Dengan cumsum, semua batas dicoba sekaligus."""
    best = (0.0, None, None)
    n = len(r)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j])
        x, rs = X[order, j], r[order]
        left_sum = np.cumsum(rs)[:-1]
        left_n = np.arange(1, n)
        # SSE turun sebanyak ini kalau dibelah setelah posisi k (rumus standar; makin besar makin bagus)
        gain = left_sum**2 / left_n + (rs.sum() - left_sum) ** 2 / (n - left_n) - rs.sum() ** 2 / n
        ok = (left_n >= min_leaf) & (n - left_n >= min_leaf) & (x[1:] != x[:-1])
        if ok.any():
            k = int(np.argmax(np.where(ok, gain, -np.inf)))
            if gain[k] > best[0]:
                best = (gain[k], j, (x[k] + x[k + 1]) / 2)
    return best


gain, j, threshold = best_split(X_train, y_train)
yes = X_train[:, j] > threshold
print(f"Pertanyaan terbaik: apakah {FEATURES[j]} > {threshold:.3f}?")
print(f"  ya   ({yes.sum():,} baris): rata-rata perubahan 14 hari {y_train[yes].mean() * 100:+.2f}%")
print(f"  tidak ({(~yes).sum():,} baris): rata-rata perubahan 14 hari {y_train[~yes].mean() * 100:+.2f}%")

# %% [markdown]
# Satu pertanyaan sudah memisahkan "akan turun" dari "akan sedikit naik". Tapi satu pertanyaan terlalu kasar.
# Decision tree yang lebih besar tinggal mengulang: di dalam kelompok "ya", cari pertanyaan terbaik lagi, dst.

# %% [markdown]
# ## 3. Boosting: banyak pertanyaan kecil yang saling memperbaiki
#
# Ide gradient boosting:
# 1. Mulai dari tebakan 0 untuk semua baris.
# 2. Hitung **residual** = y − tebakan (apa yang belum berhasil dijelaskan).
# 3. Cari split terbaik untuk menjelaskan residual itu.
# 4. Tambahkan sebagian kecil (**learning rate**, misalnya 10%) dari jawaban split itu ke tebakan.
# 5. Ulangi ratusan kali.
#
# Tiap langkah kecil memperbaiki kesalahan langkah sebelumnya. Itu seluruh rahasianya.

# %%
def stump_predict(X, j, threshold, left, right):
    return np.where(X[:, j] > threshold, right, left)


learning_rate, rounds = 0.1, 150
pred_train, pred_test = np.zeros_like(y_train), np.zeros_like(y_test)
history = []
for i in range(rounds):
    residual = y_train - pred_train
    gain, j, threshold = best_split(X_train, residual)
    if j is None:
        break
    yes = X_train[:, j] > threshold
    left, right = residual[~yes].mean(), residual[yes].mean()
    pred_train += learning_rate * stump_predict(X_train, j, threshold, left, right)
    pred_test += learning_rate * stump_predict(X_test, j, threshold, left, right)
    history.append({"tree": i + 1, "latih": mae(pred_train, y_train), "uji": mae(pred_test, y_test)})
history = pd.DataFrame(history).set_index("tree")
history.plot(figsize=(8, 3), title="Boosting buatan sendiri: error (%) per jumlah tree")
plt.axhline(mae(np.zeros_like(y_test), y_test), color="gray", ls="--", label="naive (uji)")
plt.legend()
plt.show()
history.iloc[[0, 9, 49, -1]].round(3)

# %% [markdown]
# Boosting buatan sendiri dengan 3 fitur dan pertanyaan satu tingkat sudah sedikit mengalahkan naive di data uji.
# Itu bukti idenya bekerja.

# %% [markdown]
# ## 4. LightGBM asli
#
# LightGBM melakukan hal yang sama, dengan banyak perbaikan:
# - tree-nya lebih dalam (`num_leaves`: berapa kelompok akhir per tree),
# - jauh lebih cepat (membagi nilai fitur ke ~255 "ember" dulu),
# - bisa memakai fitur kategori (provinsi, komoditas),
# - bisa mengoptimalkan target lain, misalnya kuantil (modul 5).
#
# Ayo latih dengan 3 fitur yang sama, setelan mirip proyek.

# %%
params = dict(objective="l2", learning_rate=0.05, num_leaves=31, min_child_samples=200,  # l2 = rata-rata, sama seperti boosting kita
              subsample=0.5, subsample_freq=1, seed=0, verbose=-1)
booster = lgb.train(params, lgb.Dataset(X_train, y_train), num_boost_round=300)
print(f"naive           : {mae(np.zeros_like(y_test), y_test):.3f}%")
print(f"boosting sendiri: {history['uji'].iloc[-1]:.3f}%")
print(f"LightGBM        : {mae(booster.predict(X_test), y_test):.3f}%")

# %% [markdown]
# Dengan 3 fitur, semuanya cuma sedikit lebih baik dari naive, dan selisih antar model sekecil ini masih noise.
# Sekarang tambahkan 3 fitur lagi (`ret_28`, `dev_91`, `vs_national`) tanpa mengubah modelnya.

# %%
MORE = FEATURES + ["ret_28", "dev_91", "vs_national"]
d2 = pd.DataFrame({f: model.stack(p[f].loc[mondays, cabai]) for f in MORE})
d2["y"] = model.stack((logp.shift(-14) - logp).loc[mondays, cabai])
d2 = d2.dropna().reset_index()
tr2 = d2["date"] < "2024-11-01"
b2 = lgb.train(params, lgb.Dataset(d2.loc[tr2, MORE], d2.loc[tr2, "y"]), num_boost_round=300)
print(f"naive            : {mae(np.zeros(int((~tr2).sum())), d2.loc[~tr2, 'y'].to_numpy()):.3f}%")
print(f"LightGBM 6 fitur : {mae(b2.predict(d2.loc[~tr2, MORE]), d2.loc[~tr2, 'y'].to_numpy()):.3f}%")

# %% [markdown]
# Model yang sama, fitur lebih banyak, kemenangan atas naive jadi beberapa kali lipat. Model proyek memakai
# lebih banyak lagi (14 fitur angka, provinsi, komoditas, Lebaran, semua horizon), dan di backtest 2 tahun
# menurunkan error cabai 7–11% dibanding naive. **Fitur yang bagus lebih penting daripada model yang rumit.**

# %% [markdown]
# ## 5. Overfitting: hafal soal latihan, gagal di ujian
#
# Kalau tree-nya terlalu besar dan belajarnya terlalu agresif, model mulai **menghafal noise** di data latih.
# Error latih terus turun, tapi error uji naik lagi.

# %%
curves = {}
agresif = dict(objective="l1", learning_rate=0.3, num_leaves=255, min_child_samples=5, seed=0, verbose=-1)
lgb.train(agresif, lgb.Dataset(X_train, y_train), num_boost_round=300,
          valid_sets=[lgb.Dataset(X_train, y_train), lgb.Dataset(X_test, y_test)], valid_names=["latih", "uji"],
          callbacks=[lgb.record_evaluation(curves)])
pd.DataFrame({k: np.array(v["l1"]) * 100 for k, v in curves.items()}).plot(
    figsize=(8, 3), title="Setelan agresif: error (%) per jumlah tree")
plt.axhline(mae(np.zeros_like(y_test), y_test), color="gray", ls="--")
plt.show()

# %% [markdown]
# Error latih terjun ke bawah, error uji justru memburuk, bahkan lebih jelek dari naive (garis abu-abu).
#
# Itu sebabnya setelan proyek "kalem": `learning_rate=0.05`, `min_child_samples=200` (tiap kelompok
# minimal 200 contoh, jadi tidak bisa menghafal satu-dua hari aneh), `subsample=0.5` (tiap tree cuma melihat
# setengah data acak). Dan itu sebabnya kita **selalu** menilai di data uji yang datang setelah data latih.

# %% [markdown]
# ## Ringkasan
#
# | Konsep | Satu kalimat |
# |---|---|
# | Split | Satu pertanyaan ya/tidak yang paling memisahkan data |
# | Decision tree | Pertanyaan bertingkat; tebakan = rata-rata kelompok terakhir |
# | Residual | Bagian yang belum berhasil dijelaskan |
# | Boosting | Tree kecil berturut-turut, masing-masing memperbaiki residual sebelumnya |
# | Learning rate | Seberapa besar tiap tree boleh mengubah tebakan |
# | Overfitting | Latih makin bagus, uji makin jelek |

# %% [markdown]
# ## Latihan
#
# 1. Jalankan boosting buatan sendiri dengan `learning_rate = 1.0`. Apa yang terjadi pada error uji?
# 2. Ganti `min_leaf=500` di `best_split` menjadi `min_leaf=5`, jalankan ulang boosting. Lebih bagus atau lebih jelek di uji?
# 3. Tambahkan `ret_91` dan `national_ret_7` ke `MORE` dan jalankan LightGBM 6 fitur lagi. Apakah error uji turun?
# 4. Kenapa tebakan sebuah split adalah **rata-rata** kelompoknya? (Petunjuk: angka tunggal mana yang membuat jumlah kuadrat selisih paling kecil?)

# %%
# Kerjakan di sini


# %% [markdown]
# ## Jawaban
#
# 1. Tiap tree mengubah tebakan penuh, jadi model cepat mengejar noise di data latih; error uji biasanya lebih jelek.
# 2. Kelompok sekecil 5 baris membuat split bisa "menghafal" hari-hari aneh. Biasanya uji memburuk: overfitting.
# 3. Jalankan sendiri. Kalau turun, fitur itu membawa informasi baru; kalau tidak, informasinya sudah ada di fitur lain.
# 4. Rata-rata adalah angka yang meminimalkan jumlah kuadrat selisih. Untuk error absolut, jawabannya median;
#    untuk kuantil 90%, jawabannya kuantil 90% (itu inti modul 5).
