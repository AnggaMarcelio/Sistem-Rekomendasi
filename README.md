# Laporan Proyek Machine Learning - Ch Angga Marcelio

## Domain Proyek
**Latar Belakang**

Industri penerbitan buku mengalami pertumbuhan yang pesat, dengan ribuan hingga jutaan judul baru tersedia setiap tahunnya di pasar global. Meskipun hal ini memperkaya pilihan pembaca, namun justru menimbulkan tantangan dalam memilih buku yang sesuai dengan preferensi mereka. Banyak pembaca merasa kewalahan karena banyaknya pilihan yang ada, sehingga tidak jarang melewatkan buku-buku yang sebenarnya relevan dan menarik bagi mereka [[1](https://www.mdpi.com/2076-3417/13/11/6833)].

Di sisi lain, penulis dan penerbit menghadapi kesulitan dalam menjangkau pembaca yang tepat di tengah persaingan pasar yang semakin ketat. Strategi pemasaran konvensional tidak selalu efektif untuk menyampaikan karya kepada audiens yang sesuai. Oleh karena itu, diperlukan pendekatan berbasis teknologi yang mampu menyajikan rekomendasi personalisasi yang lebih akurat kepada pengguna. Sistem rekomendasi hadir sebagai solusi untuk menjembatani kebutuhan pembaca dan pelaku industri buku [[2](https://www.sciencedirect.com/science/article/pii/S2090447923001521)].

Dalam konteks ini, metode Collaborative Filtering berbasis neural network dan embedding semakin populer karena kemampuannya mempelajari representasi laten pengguna dan item secara lebih mendalam. Pendekatan ini terbukti mampu meningkatkan akurasi rekomendasi, terutama dalam menghadapi masalah data yang jarang (sparse data) atau cold-start. Dengan mengintegrasikan metadata dan representasi fitur ke dalam arsitektur jaringan saraf, sistem rekomendasi dapat memberikan hasil yang lebih relevan dan personal bagi setiap pengguna [[3](https://journal.ugm.ac.id/ijccs/article/view/103611)].

**Mengapa dan bagaimana masalah ini harus diselesaikan?**

Masalah dalam menemukan buku yang relevan di tengah lautan pilihan yang tersedia harus diselesaikan karena berdampak langsung pada kepuasan pengguna, efektivitas distribusi konten, dan keberlanjutan industri buku itu sendiri. Ketika pengguna kesulitan menemukan buku yang sesuai dengan preferensi mereka, kemungkinan besar mereka akan kehilangan minat atau bahkan berpindah ke platform lain yang menawarkan pengalaman lebih personal dan efisien. Bagi penerbit dan penulis, kegagalan dalam menjangkau audiens yang tepat dapat menghambat visibilitas karya mereka dan menurunkan potensi pendapatan.

Masalah ini dapat diselesaikan dengan mengimplementasikan sistem rekomendasi berbasis Collaborative Filtering yang ditingkatkan dengan metode Neural Network Embedding. Pendekatan ini memungkinkan sistem memahami pola tersembunyi dalam interaksi pengguna-buku, serta merepresentasikan karakteristik pengguna dan buku dalam bentuk vektor yang dapat dibandingkan. Dengan memanfaatkan teknik ini, sistem dapat menyarankan buku yang relevan bahkan ketika data yang tersedia masih terbatas atau pengguna baru pertama kali menggunakan platform. Hal ini meningkatkan pengalaman personalisasi, mempercepat proses pencarian buku, serta membantu penerbit dan penulis menjangkau segmen audiens yang lebih tepat dan potensial.

## Business Understanding
### Problem Statements

Permasalahan utama yang melatarbelakangi proyek ini adalah:
- Pengguna seringkali kesulitan menemukan buku yang sesuai dengan minat mereka dari jutaan pilihan yang tersedia.
- Kurangnya personalisasi dalam penemuan buku dapat menyebabkan pengguna merasa overwhelmed atau melewatkan buku-buku potensial yang akan mereka sukai.

### Goals

Adapun tujuan yang ingin dicapai dalam proyek ini adalah:
- Membangun sistem rekomendasi buku yang efektif dan efisien.
- Memberikan rekomendasi buku yang relevan dan personal kepada pengguna berdasarkan riwayat rating dan preferensi mereka.

### Solution Approach

Untuk mencapai tujuan ini, akan menggunakan pendekatan Collaborative Filtering dengan implementasi Neural Network (Embedding). Pendekatan ini dipilih karena dataset yang tersedia memiliki data rating eksplisit dari pengguna terhadap buku, yang sangat ideal untuk melatih model collaborative filtering. Model ini akan belajar pola preferensi dari interaksi pengguna-item (rating) dan merekomendasikan buku berdasarkan kesamaan preferensi antar pengguna atau kesamaan karakteristik rating antar buku. Neural Network dengan embedding layer dipilih karena kemampuannya untuk menangkap pola kompleks dan skalabilitas yang baik.

## Data Understanding

Tahap ini bertujuan untuk memahami struktur, karakteristik, dan kualitas data yang akan digunakan serta memuat dataset dan melakukan analisis deskriptif serta eksplorasi data awal.

Dataset yang digunakan adalah Book-Crossing Dataset, yang dapat diunduh dari Kaggle, "Book-Recommendation-Dataset," Kaggle, 2021. [Online]. Tersedia: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset. 

### Struktur Dataset

Dataset ini terdiri dari tiga file CSV:

- `Books.csv`: Berisi informasi detail tentang buku. Sebelum pra-pemrosesan, DataFrame ini memiliki 271.360 entri.
- `Users.csv`: Berisi informasi detail tentang pengguna. Sebelum pra-pemrosesan, DataFrame ini memiliki 278.858 entri.
- `Ratings.csv`: Berisi data rating yang diberikan pengguna untuk buku. Sebelum pra-pemrosesan, DataFrame ini memiliki 1.149.780 entri.

### Variabel-variabel pada dataset adalah sebagai berikut:

1. df_books
- `ISBN`: International Standard Book Number, pengidentifikasi unik untuk buku.
- `Book-Title`: Judul buku.
- `Book-Author`: Nama penulis buku.
- `Year-Of-Publication`: Tahun publikasi buku.
- `Publisher`: Penerbit buku.
- `Image-URL-S, Image-URL-M, Image-URL-L`: URL gambar sampul buku dalam berbagai ukuran (Small, Medium, Large).

2. df_users
- `User-ID`: Pengidentifikasi unik untuk pengguna.
- `Location`: Lokasi pengguna (kota, negara bagian, negara).
- `Age`: Usia pengguna.

3. df_ratings
- `User-ID`: Pengidentifikasi pengguna yang memberikan rating.
- `ISBN`: ISBN dari buku yang di-rating.
- `Book-Rating`: Rating yang diberikan pengguna untuk buku, dalam skala 0-10.

**Exploratory Data Analysis (EDA)**:

Gambar 1. Distribusi Rating Buku (Plot)
![Visualisasi Distribusi Rating Buku](./image/Distribusi%20Rating%20Buku.png)

Insight: Visualisasi ini secara jelas menunjukkan bahwa rating 0 harus ditangani secara terpisah atau dihapus untuk fokus pada preferensi eksplisit.

Gambar 2. Distribusi Usia Pengguna (Plot)
![Visualisasi Distribusi Usia Pengguna](./image/Distibusi%20Usia%20Pengguna.png)

Insight: Plot ini mengidentifikasi demografi pengguna utama dan mendukung kebutuhan untuk membersihkan data usia yang tidak valid.

Gambar 3. Distribusi Top 10 Penulis Terpopuler (Plot)
![Visualisasi Distribusi Top 10 Penulis Terpopuler](./image/Distribusi%20Penulis.png)

Insight: Visualisasi ini menegaskan fenomena "long-tail" dalam data, di mana beberapa penulis sangat populer.

## Data Preparation

Pada bagian ini, menerapkan berbagai teknik data preparation untuk memastikan data bersih, konsisten, dan siap untuk tahap modeling.

**Teknik Data Preparation**

1. Penanganan Kolom 'Year-Of-Publication' pada df_books: Kolom ini diubah dari tipe object menjadi numerik. Nilai-nilai yang tidak masuk akal (seperti tahun di masa depan atau tahun yang terlalu lampau) diubah menjadi NaN kemudian diisi dengan nilai modus (tahun publikasi yang paling sering muncul).
2. Penanganan Missing Values pada df_books dan df_users:
- Untuk df_books, baris dengan missing values pada kolom Book-Author dan Publisher dihapus.
- Untuk df_users, NaN pada kolom Age diisi dengan median usia. Usia yang tidak masuk akal (< 5 atau > 100 tahun) juga diperbaiki dengan median usia, dan kolom Age dikonversi ke tipe integer.
3. Penanganan Rating 0 di df_ratings (Filtering Rating Implisit): Semua rating yang bernilai 0 dihapus dari dataset, menghasilkan df_ratings_explicit. Jumlah rating berkurang dari 1.149.780 menjadi 433.671 setelah filter.
4. Penggabungan DataFrame: df_ratings_explicit digabungkan dengan df_books berdasarkan ISBN, dan hasilnya kemudian digabungkan dengan df_users berdasarkan User-ID, menghasilkan final_df dengan 383.837 entri.
5. Mengganti Nama Kolom untuk Konsistensi: Kolom User-ID, ISBN, dan Book-Rating di final_df diganti namanya menjadi user_id, item_id, dan rating.

**Alasan Tahapan Data Preparation Dilakukan**

1. Validasi dan Konsistensi Data: Mengonversi Year-Of-Publication ke tipe numerik dan membersihkan nilai-nilai ekstrem memastikan data akurat dan konsisten untuk analisis.
2. Integritas dan Kualitas Data: Penanganan missing values pada Book-Author, Publisher, dan Age mencegah error pada model dan memastikan bahwa fitur-fitur ini dapat digunakan secara efektif tanpa bias. Median dipilih untuk usia karena robust terhadap outlier.
3. Fokus pada Preferensi Eksplisit: Penghapusan rating 0 sangat krusial karena model yang dibangun berfokus pada rating eksplisit, yang merupakan indikasi jelas preferensi pengguna. Rating 0 dapat mendistorsi pola preferensi jika tidak dihapus.
4. Analisis Holistik dan Basis Pemodelan: Penggabungan DataFrame memungkinkan akses data yang komprehensif, yang esensial untuk membangun model Collaborative Filtering yang memerlukan hubungan antara pengguna dan item.
5. Standardisasi dan Kompatibilitas: Penggantian nama kolom membantu dalam standardisasi penamaan dan keterbacaan kode, serta mempermudah integrasi dengan library machine learning yang mungkin memiliki persyaratan penamaan tertentu.

## Modeling and Result

Pada tahap ini, membangun sistem rekomendasi menggunakan pendekatan Collaborative Filtering dengan implementasi Neural Network (Embedding) dan akan menyajikan top-N recommendation sebagai output.

## Evaluation

### Metrik Evaluasi yang Digunakan


**Referensi**

Dataset: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

[1] H. Kim, "Feature Extracted Deep Neural Collaborative Filtering for E-Book Service Recommendations," Applied Sciences, vol. 13, no. 11, 2023. [Online]. Tersedia: https://www.mdpi.com/2076-3417/13/11/6833

[2] M. Rezaei and M. Jalili, "Knowledge Graph-Based Recommendation System Enhanced by Neural Collaborative Filtering and Knowledge Graph Embedding," Ain Shams Engineering Journal, vol. 15, no. 1, 2024. [Online]. Tersedia: https://www.sciencedirect.com/science/article/pii/S2090447923001521

[3] P. A. S. Mukti and Z. K. A. Baizal, "Enhancing Neural Collaborative Filtering with Metadata for Book Recommender System," Indonesian Journal of Computing and Cybernetics Systems, vol. 19, no. 1, 2025. [Online]. Tersedia: https://journal.ugm.ac.id/ijccs/article/view/103611
