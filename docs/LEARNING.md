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
