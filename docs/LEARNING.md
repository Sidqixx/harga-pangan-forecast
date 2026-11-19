# Catatan Belajar

Panduan belajar proyek ini, diisi bertahap seiring proyek jalan. Tiap bagian: apa yang dibikin, konsepnya, dan pertanyaan interview yang harus bisa dijawab.

## Rilis 1: pipeline data

### `fetch.py`: ambil data
- **Sumber:** endpoint JSON di balik tabel PIHPS (`GetGridDataKomoditas`). Ketemu dengan baca kode JavaScript halaman web-nya, bukan dari dokumentasi resmi.
- **Format lebar ke panjang:** API kasih 1 baris per provinsi dengan 1 kolom per tanggal. Kita ubah jadi 1 baris per (tanggal, provinsi, komoditas). Format panjang gampang di-filter, di-join, dan di-append.
- **Idempotent + upsert:** key unik `(date, province, commodity_id)`. Data baru menimpa data lama dengan key yang sama, jadi script aman dijalankan berkali-kali dan revisi harga dari BI otomatis masuk.
- **Retry dengan exponential backoff:** server PIHPS sering memutus koneksi. Tunggu 5, 10, 20, 40, 80 detik sebelum coba lagi, karena server yang lagi sibuk butuh jeda yang makin panjang.
- **Kegagalan parsial:** satu chunk gagal tidak menghentikan yang lain. Di akhir, script exit 1 (supaya cron merah) dan mencetak perintah untuk mengulang.

### `quality.py`: cek kualitas
- **Median terpusat (centered):** harga dibandingkan dengan median 21 observasi di sekitarnya. Typo itu sekali muncul lalu hilang, sedangkan lonjakan asli bertahan, sehingga median ikut naik dan lonjakannya lolos. Lonjakan asli adalah hal yang mau diprediksi, jadi tidak boleh ikut terbuang.
- **Data basi:** kalau harga terbaru lebih tua dari 10 hari, job gagal dan GitHub mengirim email. Batasnya 10 hari, bukan 3, karena cuti bersama Lebaran.

### Cron (GitHub Actions)
- Jadwal ditulis dalam UTC: `0 3 * * *` = 10:00 WIB.
- Job `update` menarik data dan commit hanya kalau ada perubahan. Job `dashboard` membangun ulang halaman web dari data terbaru.
- `if: always()` = langkah tetap jalan walaupun langkah sebelumnya gagal, jadi data yang sudah berhasil ditarik tetap disimpan.

**Pertanyaan interview**
1. Apa yang terjadi kalau situs PIHPS down 3 hari? *(Job merah tiap hari, email masuk, data lama tetap aman. Saat situs hidup lagi, jendela 14 hari otomatis menutup bolongnya.)*
2. Kenapa datanya CSV per bulan, bukan satu file? *(Cron cuma menulis ulang file bulan berjalan, jadi diff di git kecil.)*
3. Kenapa tidak pakai Spark atau warehouse? *(725 ribu baris = 44 MB. Satu laptop sudah lebih dari cukup.)*

## Rilis 2: forecast

### `backtest.py`: cara menguji forecaster
- **Rolling-origin backtest:** pilih tanggal "origin" (tiap Senin, 2 tahun terakhir, 104 kali). Di tiap origin, forecaster cuma boleh melihat data sampai hari itu, lalu menebak 14 hari ke depan. Tebakannya dibandingkan dengan harga asli. Ini mensimulasikan cara model dipakai di dunia nyata: hari ini menebak masa depan.
- **Kenapa bukan random split / k-fold acak:** kalau data dikocok, model bisa belajar dari hari Rabu untuk menebak hari Selasa. Skornya kelihatan bagus, tapi di produksi tidak ada data masa depan. Itu namanya **leakage**.
- **Anti-leakage by construction:** harness hanya mengirim `known.loc[:origin]` ke forecaster. Forecaster secara fisik tidak bisa melihat data setelah origin, jadi kita tidak bergantung pada kedisiplinan tiap forecaster.
- **Horizon:** jarak hari dari origin ke tanggal yang ditebak (1 sampai 14). Makin jauh, makin susah, jadi error dilaporkan per kelompok horizon (1-3, 4-7, 8-14 hari).
- **MAPE:** rata-rata `|tebakan - asli| / asli`, dalam persen. Dipakai karena harga cabai (Rp 50 ribuan) dan beras (Rp 13 ribuan) beda skala, dan persen bisa dibandingkan antar komoditas.

### Baseline: batas minimal yang harus dikalahkan
| Baseline | Cara kerja |
|---|---|
| `naive` | Besok sampai 14 hari ke depan = harga terakhir |
| `seasonal` | Harga terakhir × perubahan harga di 14 hari yang sama tahun lalu |

**Hasil (104 origin, Nov 2024 – Nov 2026, 314.730 tebakan):** `naive` mengalahkan `seasonal` di semua komoditas dan semua horizon. Secara keseluruhan, MAPE naive 2,1% / 4,0% / 6,3% untuk horizon 1-3 / 4-7 / 8-14 hari, sedangkan seasonal 3,4% / 6,3% / 9,7%. Artinya pergerakan harga tahun lalu tidak berulang dengan tanggal yang sama, jadi meniru tahun lalu malah menambah noise. Pola musiman yang kelihatan di dashboard itu rata-rata banyak tahun, bukan pola yang terulang persis tiap tahun.

Yang paling susah: cabai rawit merah, 13,6% di horizon 8-14 hari. Yang paling gampang: beras, 0,3%.

**Pertanyaan interview**
1. Kenapa tidak pakai random split? *(Leakage, lihat di atas.)*
2. Kenapa perlu baseline? *(Tanpa baseline, MAPE 5% tidak ada artinya. Kalau "harga kemarin" sudah dapat 4%, model dengan 5% itu justru lebih buruk.)*
3. Baseline mana yang harus dikalahkan model? *(Naive. Itu yang terkuat, jadi itu standarnya.)*

### `model.py`: LightGBM kuantil
- **Satu model global:** satu model untuk 304 deret sekaligus, bukan satu model per deret. Deret yang datanya sedikit (misalnya cabai di provinsi kecil) ikut belajar dari pola deret lain. Komoditas dan provinsi jadi fitur supaya model tetap tahu bedanya.
- **Yang ditebak adalah perubahan, bukan harga:** target = `log(harga nanti / harga terakhir)`. Beras Rp 13 ribu naik Rp 1.300 dan cabai Rp 70 ribu naik Rp 7.000 itu sama-sama naik ~10%, jadi satu model bisa belajar dari keduanya. Hasilnya dikonversi balik ke rupiah.
- **Fitur (semuanya dihitung dari data sampai hari origin):**
  - `ret_k`: perubahan harga k hari terakhir (momentum)
  - `vol_28`: seberapa liar harga 4 minggu terakhir
  - `dev_28`, `dev_91`: harga sekarang vs median 1 dan 3 bulan. Harga yang sudah jauh di atas median cenderung turun lagi (*mean reversion*)
  - `vs_national`: provinsi ini lebih mahal atau lebih murah dari nasional
  - `lebaran_until`, `lebaran_since`: jarak ke Lebaran, dibatasi maksimal 60 hari
- **Dilatih ulang di dalam backtest** (tiap 6 bulan), dan hanya pakai contoh yang harga aslinya sudah keluar sebelum origin. Kalau model dilatih sekali pakai semua data lalu diuji di 2 tahun terakhir, model sudah "melihat" jawaban ujiannya.

### P10, P50, P90 dan coverage
Contoh nyata: forecast cabai rawit merah DKI Jakarta dari data 19 Nov 2026 (harga terakhir Rp 76.650).

| Tanggal | P10 | P50 | P90 |
|---|---|---|---|
| 20 Nov (1 hari) | Rp 75.308 | Rp 76.878 | Rp 80.123 |
| 26 Nov (7 hari) | Rp 70.826 | Rp 76.780 | Rp 90.593 |
| 3 Des (14 hari) | Rp 67.519 | Rp 77.608 | Rp 104.650 |

Cara bacanya:
- **P50** = tebakan tengah. Harga kemungkinan besar tetap di sekitar Rp 77 ribu.
- **P10 dan P90** = batas bawah dan atas. Model "80% yakin" harga 3 Desember ada di antara Rp 67,5 ribu dan Rp 104,6 ribu.
- **Rentangnya melebar** makin jauh ke depan, karena masa depan yang jauh lebih tidak pasti.
- **Rentangnya miring ke atas:** P90 (+36%) jauh lebih jauh dari P50 dibanding P10 (−13%). Model belajar bahwa cabai lebih sering melonjak tajam daripada anjlok tajam. Informasi "risiko naiknya besar" ini yang nanti dipakai untuk alert.

**Coverage** = dari semua tebakan di backtest, berapa persen harga asli yang benar-benar jatuh di antara P10 dan P90. Kalau model jujur, harusnya sekitar 80%.
- Hasil keseluruhan: 78,5–80,5%. Rentangnya jujur.
- Cabai 8-14 hari: 73–78%. Sedikit terlalu pede, karena lonjakan cabai kadang lebih besar dari yang model kira.

### Hasil backtest
| Horizon | naive | seasonal | LightGBM |
|---|---|---|---|
| 1-3 hari | 2,1% | 3,4% | **2,0%** |
| 4-7 hari | 4,0% | 6,3% | **3,7%** |
| 8-14 hari | 6,3% | 9,7% | **5,8%** |

LightGBM tidak pernah kalah dari naive, dan menang paling besar di bawang merah (−16% error) dan cabai rawit merah (−11%). Di beras dan bawang putih hasilnya seri, karena harganya sudah hampir tidak bergerak dan tidak ada yang bisa dimenangkan.

**Pertanyaan interview**
1. Kenapa satu model global, bukan model per deret? *(Deret kecil ikut belajar dari deret besar, dan cuma perlu merawat 3 model, bukan 900.)*
2. Kenapa targetnya log-perubahan? *(Supaya semua komoditas satu skala. Error 10% di beras dan cabai dihitung sama.)*
3. Model cuma menang 8% dari naive. Berguna tidak? *(Kemenangannya ada di komoditas yang paling penting dan paling liar. Selain itu model memberi rentang risiko P10–P90 yang jujur, sesuatu yang naive tidak punya.)*
4. Kenapa coverage dicek? *(Rentang yang terlalu sempit bikin user kaget, rentang yang terlalu lebar tidak berguna. Coverage membuktikan rentangnya bisa dipercaya.)*

### `live.py`: forecast harian live
- **Tiap hari cron:** latih ulang model pakai semua data, tebak 14 hari ke depan untuk 304 deret, simpan ke `data/forecasts/`. Isinya P10, P50, P90, dan naive (biar bisa dibandingkan).
- **Penilaian otomatis:** begitu harga asli untuk tanggal yang ditebak keluar, dashboard mencocokkan tebakan lama dengan harga asli dan menghitung error-nya. Ini **delayed labels**: jawaban untuk tebakan hari ini baru keluar 1 sampai 14 hari kemudian.
- **Kenapa ini penting:** backtest itu simulasi masa lalu. Akurasi live itu bukti bahwa model benar-benar jalan di dunia nyata, dengan data yang belum pernah ia lihat. Kalau akurasi live jauh lebih jelek dari backtest, berarti ada yang salah (misalnya leakage di backtest, atau pola harga berubah).
- **Idempotent lagi:** cron yang jalan dua kali di hari yang sama (atau di akhir pekan, saat harga baru belum ada) menimpa tebakan untuk origin yang sama, bukan menumpuk.

**Pertanyaan interview**
1. Gimana kamu tahu model rusak di produksi kalau labelnya baru datang 2 minggu kemudian? *(Horizon 1 hari sudah bisa dinilai besoknya, jadi sinyal paling awal datang dalam sehari. Coverage P10–P90 yang turun jauh dari 80% juga tanda bahaya.)*
2. Kenapa simpan naive juga? *(Supaya akurasi live model selalu dibandingkan dengan baseline di hari yang sama. Error 5% di bulan Lebaran bisa jadi bagus kalau naive 8%.)*

## Rilis 3: peringatan dini lonjakan

### `spike.py`: definisi "lonjakan"
- **Batas waspada** = median 20 laporan terakhir (~4 minggu) × (1 + ambang). Ambang beda per komoditas: beras 5%, ayam/telur/bawang putih 10%, bawang merah 20%, cabai rawit hijau 30%, cabai merah 40%, cabai rawit merah 50%.
- **Cara milih ambangnya:** dari data 2017–2024 (sebelum periode tes, biar tidak curang), dipilih ambang yang bikin lonjakan baru terjadi di ~10% hari. Kalau pakai 10% untuk semua, cabai "waspada" hampir tiap hari (alert yang terus bunyi akhirnya diabaikan), dan beras hampir tidak pernah.
- **Yang ditebak cuma lonjakan BARU:** deret yang hari ini sudah di atas batas sudah kelihatan di tabel waspada, tidak perlu ditebak. Nilai sebuah peringatan dini ada di deret yang sekarang masih tenang.
- **Label:** 1 kalau dalam 14 hari ke depan ada harga yang menembus batas, 0 kalau tidak. Label hari ini baru diketahui 14 hari lagi.

### Classifier dan cara menilainya
- **Model:** LightGBM klasifikasi (`objective="binary"`), keluarannya **peluang** 0–100%.
- **Kenapa tidak pakai akurasi:** cuma 7,5% hari yang jadi lonjakan. Model yang selalu bilang "tidak akan melonjak" sudah 92,5% akurat tapi tidak berguna sama sekali.
- **Precision** = dari semua peringatan, berapa persen yang benar-benar melonjak. **Recall** = dari semua lonjakan, berapa persen yang sempat diperingatkan. Dua-duanya saling tarik-menarik.
- **Average precision (AP)** = ringkasan precision di semua tingkat recall. Model acak dapat AP = 7,5% (sama dengan frekuensi lonjakan).

| | AP |
|---|---|
| Acak | 0,075 |
| Heuristik: "yang harganya sudah paling dekat ke batas" | 0,193 |
| **Classifier** | **0,490** |

- **Cut-off peluang:** dashboard pakai ≥ 50%, jadi ~9 peringatan per hari, 65% benar, menangkap 27% lonjakan. Cut-off 70% artinya 2,7 peringatan per hari, 81% benar, tapi cuma menangkap 10%. Ini **keputusan bisnis, bukan keputusan teknis**: user mau daftar pendek yang hampir pasti benar, atau daftar panjang yang menangkap lebih banyak?

### Tes anti-leakage
Di backtest spike, fitur dihitung sekali untuk seluruh data supaya cepat. Itu cuma aman kalau fitur di tanggal X tidak berubah saat data setelah X ditambahkan. `tests/test_spike.py` membuktikan ini: fitur dihitung dari data penuh vs data yang dipotong di tanggal X, hasilnya harus sama persis.

### Gerbang rilis (`live.py`)
Setiap hari, sebelum model baru dipublikasikan:
1. Latih ulang dengan resep yang sama, **seolah-olah 4 minggu lalu**.
2. Uji di 4 minggu terakhir yang tidak pernah dilihat model.
3. Lolos hanya kalau: forecast tidak kalah dari naive (toleransi 2%), rentang P10–P90 memuat 70–90% harga asli, dan AP classifier mengalahkan heuristik.
4. Kalau gagal: tidak ada yang dipublikasikan, forecast terakhir yang lolos tetap tampil, job merah, email masuk.

Kenapa perlu: data bisa rusak, kode bisa berubah, pola harga bisa bergeser. Tanpa gerbang, model rusak langsung tayang ke user.

### `MODEL_CARD.md`
Dokumen standar di industri: model ini untuk apa, **tidak** untuk apa, datanya apa, hasil evaluasinya, dan keterbatasannya. Yang paling sering ditanya interviewer justru bagian keterbatasan.

**Pertanyaan interview**
1. Kenapa ambangnya beda per komoditas? *(Cabai wajar naik 30% dalam 2 minggu, beras naik 5% sudah luar biasa. Satu ambang = alert cabai bunyi terus, alert beras tidak pernah.)*
2. Kenapa AP, bukan akurasi? *(Kelas tidak seimbang, 7,5% positif. Akurasi bisa tinggi dengan model yang tidak berguna.)*
3. Model kamu benar 65%. Itu bagus? *(Dibanding acak 7,5% dan heuristik 19%, jauh lebih baik. Dan cut-off bisa digeser kalau user butuh lebih yakin.)*
4. Apa yang terjadi kalau besok modelnya rusak? *(Gerbang rilis gagal, tidak ada yang dipublikasikan, versi terakhir yang lolos tetap tampil, dan saya dapat email.)*
5. Apa risiko etis sistem ini? *(Peringatan publik bisa memicu penimbunan. Karena itu peringatan ditampilkan bersama track record-nya, sebagai peluang, dan ditujukan ke pengelola pasokan.)*
