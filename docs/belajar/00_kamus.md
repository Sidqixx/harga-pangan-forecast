# Kamus istilah

Semua istilah di proyek ini, dijelaskan dengan bahasa biasa dan contoh dari data harga pangan. Diurutkan dari yang paling dasar.

## Data

| Istilah | Artinya | Contoh di proyek |
|---|---|---|
| **Deret waktu** (*time series*) | Satu nilai yang dicatat berulang dari waktu ke waktu | Harga cabai rawit merah di DKI tiap hari sejak 2017 |
| **Deret** (*series*) | Satu kombinasi yang punya riwayatnya sendiri | Proyek ini punya 304 deret aktif = 9 komoditas × ~34 provinsi |
| **Format panjang** (*long*) | Satu baris = satu pengamatan | `2026-11-19, DKI Jakarta, Cabai Rawit Merah, 76650` |
| **Format lebar** (*wide*) | Satu baris = satu tanggal, satu kolom = satu deret | Baris 19 Nov, kolom "Cabai Rawit Merah DKI" = 76650 |
| **Data bolong** (*missing*) | Hari tanpa laporan | Sabtu, Minggu, libur, provinsi yang lupa lapor |
| **Forward fill** (`ffill`) | Isi hari bolong dengan nilai terakhir yang diketahui | Harga Jumat dipakai untuk Sabtu dan Minggu, maksimal 10 hari |
| **Outlier** | Nilai yang tidak masuk akal dibanding sekitarnya | Bawang merah Rp 1.200 padahal biasanya Rp 25.000 (salah ketik) |
| **Median** | Nilai tengah setelah diurutkan. Tidak terpengaruh nilai ekstrem, tidak seperti rata-rata | Median 20 laporan terakhir dipakai sebagai harga "normal" |
| **Idempotent** | Dijalankan sekali atau sepuluh kali, hasilnya sama | `fetch.py` dua kali tidak membuat data dobel |
| **Upsert** | Update kalau sudah ada, insert kalau belum | Harga yang direvisi BI menimpa harga lama dengan tanggal yang sama |

## Forecast

| Istilah | Artinya | Contoh |
|---|---|---|
| **Forecast** | Tebakan nilai di masa depan | "Cabai DKI tanggal 3 Des: Rp 77.608" |
| **Origin** | Hari tebakan dibuat. Model cuma boleh tahu data sampai hari ini | Origin 19 Nov |
| **Horizon** (*h*) | Berapa hari ke depan dari origin | 20 Nov = h1, 3 Des = h14 |
| **Target** | Nilai yang mau ditebak | Harga 14 hari lagi |
| **Fitur** (*feature*) | Informasi yang dipakai model untuk menebak | Perubahan harga 7 hari terakhir, jarak ke Lebaran |
| **Label** | Jawaban yang benar untuk dipelajari model | Harga asli yang keluar belakangan |
| **Baseline** | Cara menebak yang paling sederhana. Model harus mengalahkannya supaya ada gunanya | `naive` = harga besok sama dengan harga terakhir |
| **Seasonal naive** | Baseline yang meniru pola tahun lalu | Kalau tahun lalu naik 10% di periode ini, tebak naik 10% |
| **Error** | Selisih tebakan dan harga asli | Tebak 70.000, asli 77.000, error 7.000 |
| **MAPE** | Rata-rata error dalam persen dari harga asli | 7.000 / 77.000 = 9,1% |
| **Backtest** | Menguji cara menebak di masa lalu, seolah-olah waktu diputar mundur | Pura-pura hari ini 4 Nov 2024, tebak, cek; lalu 11 Nov 2024; dst. |
| **Rolling-origin** | Backtest dengan banyak origin yang terus bergeser maju | 104 origin, tiap Senin selama 2 tahun |
| **Leakage** | Model tidak sengaja melihat data masa depan, sehingga skornya bagus palsu | Fitur yang menghitung rata-rata 10 hari *ke depan* |

## Fitur

| Istilah | Artinya | Contoh |
|---|---|---|
| **Log return** | Perubahan harga dalam skala log. Naik 10% lalu turun 10% kembali tepat ke nol | `log(harga hari ini / harga 7 hari lalu)` |
| **Momentum** | Kecenderungan harga yang sedang naik untuk terus naik sebentar | `ret_7` positif |
| **Volatilitas** | Seberapa liar harga bergerak | Cabai 1,18 per tahun, beras 0,34 |
| **Mean reversion** | Harga yang sudah jauh dari normal cenderung kembali | Cabai yang sudah ~40% di atas median 28 hari rata-rata turun dalam 28 hari ke depan (modul 3) |
| **Fitur kategori** | Fitur berupa nama, bukan angka | Komoditas, provinsi |
| **Model global** | Satu model untuk semua deret | Satu LightGBM untuk 304 deret |

## Model

| Istilah | Artinya |
|---|---|
| **Decision tree** | Serangkaian pertanyaan ya/tidak ("harga sudah 20% di atas median?") yang berakhir di sebuah tebakan |
| **Split** | Satu pertanyaan di tree, dipilih yang paling memisahkan data |
| **Residual** | Sisa error yang belum bisa dijelaskan model |
| **Gradient boosting** | Banyak tree kecil disusun berurutan; tiap tree baru belajar memperbaiki residual tree-tree sebelumnya |
| **LightGBM** | Library gradient boosting yang cepat, dipakai luas di industri untuk data tabel |
| **Hyperparameter** | Setelan model yang kita pilih sendiri, bukan dipelajari dari data: jumlah tree, ukuran tree, learning rate |
| **Learning rate** | Seberapa besar tiap tree baru boleh mengubah tebakan. Kecil = pelan tapi stabil |
| **Overfitting** | Model hafal data latih tapi jelek di data baru |
| **Train / test** | Data untuk belajar / data untuk menguji. Untuk deret waktu, test harus selalu *setelah* train |

## Kuantil dan ketidakpastian

| Istilah | Artinya | Contoh |
|---|---|---|
| **Kuantil** | Nilai yang membagi data. P10 = 10% data di bawahnya | P10 harga = batas bawah yang wajar |
| **P10 / P50 / P90** | Batas bawah / tebakan tengah / batas atas | Rp 67,5 rb / 77,6 rb / 104,7 rb |
| **Rentang 80%** | Antara P10 dan P90; harga asli harusnya ada di dalamnya 80% waktu | |
| **Quantile regression** | Model yang langsung menebak kuantil, bukan rata-rata | LightGBM dengan `objective="quantile"` |
| **Pinball loss** | Cara menghukum tebakan kuantil. Untuk P90, menebak terlalu rendah dihukum 9× lebih berat | |
| **Coverage** | Persen harga asli yang jatuh di dalam rentang | Target 80% |
| **Kalibrasi** | Model bilang 80% yakin, dan benar 80% | Coverage 78,5–80,5% = terkalibrasi |

## Klasifikasi

| Istilah | Artinya | Contoh |
|---|---|---|
| **Klasifikasi** | Menebak kategori (ya/tidak), bukan angka | Akan melonjak dalam 14 hari? |
| **Probabilitas** | Keyakinan model, 0 sampai 1 | 0,66 = 66% |
| **Base rate** | Seberapa sering kejadiannya secara alami | 7,5% hari berubah jadi lonjakan |
| **Kelas tidak seimbang** | Satu jawaban jauh lebih jarang dari yang lain | 7,5% "ya" vs 92,5% "tidak" |
| **Cut-off** / threshold | Batas probabilitas untuk menyalakan alert | Peluang ≥ 50% = peringatan |
| **True positive (TP)** | Diperingatkan, dan memang melonjak | |
| **False positive (FP)** | Diperingatkan, ternyata tidak melonjak (alarm palsu) | |
| **False negative (FN)** | Tidak diperingatkan, ternyata melonjak (kecolongan) | |
| **Precision** | Dari semua peringatan, berapa persen benar = TP / (TP + FP) | 65% |
| **Recall** | Dari semua lonjakan, berapa persen sempat diperingatkan = TP / (TP + FN) | 27% |
| **Average precision (AP)** | Ringkasan precision di semua kemungkinan cut-off. Acak = base rate | 0,49 vs acak 0,075 |
| **Heuristik** | Aturan sederhana buatan manusia, tanpa machine learning | "Yang harganya paling dekat ke batas" |

## Produksi (MLOps)

| Istilah | Artinya | Di proyek |
|---|---|---|
| **Pipeline** | Urutan langkah otomatis dari data mentah sampai hasil | fetch, quality, live, report |
| **Cron** | Jadwal otomatis | Tiap hari 10:17 WIB |
| **CI** (*continuous integration*) | Tes otomatis setiap kali kode berubah atau job jalan | `pytest` di awal job |
| **Delayed labels** | Jawaban benar baru datang belakangan | Tebakan h14 baru bisa dinilai 14 hari kemudian |
| **Monitoring** | Memantau apakah model masih bagus setelah dipakai | Tabel akurasi live di dashboard |
| **Drift** | Pola data berubah dari waktu model dilatih | Harga cabai tahun ini jauh lebih liar |
| **Gerbang rilis** (*release gate*) | Ujian otomatis sebelum model baru boleh dipakai | Harus lolos 3 syarat di 4 minggu terakhir |
| **Champion / challenger** | Model lama (juara) vs model baru (penantang); yang baru dipakai hanya kalau menang | Di proyek ini penantangnya dibandingkan dengan naive dan heuristik |
| **Fallback** | Rencana cadangan kalau ada yang gagal | Gerbang gagal = hasil terakhir yang lolos tetap tampil |
| **Model card** | Dokumen resmi model: untuk apa, datanya, hasil, keterbatasan | `MODEL_CARD.md` |
| **Deterministik** | Input sama selalu menghasilkan output sama | `seed=0` di LightGBM |
