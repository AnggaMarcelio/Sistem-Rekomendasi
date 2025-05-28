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

Untuk mencapai tujuan ini, kami akan menggunakan pendekatan Collaborative Filtering dengan implementasi Neural Network (Embedding). Pendekatan ini dipilih karena dataset yang tersedia memiliki data rating eksplisit dari pengguna terhadap buku, yang sangat ideal untuk melatih model collaborative filtering. Model ini akan belajar pola preferensi dari interaksi pengguna-item (rating) dan merekomendasikan buku berdasarkan kesamaan preferensi antar pengguna atau kesamaan karakteristik rating antar buku. Neural Network dengan embedding layer dipilih karena kemampuannya untuk menangkap pola kompleks dan skalabilitas yang baik.

## Data Understanding

**Referensi**

Dataset: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

[1] H. Kim, "Feature Extracted Deep Neural Collaborative Filtering for E-Book Service Recommendations," Applied Sciences, vol. 13, no. 11, 2023. [Online]. Tersedia: https://www.mdpi.com/2076-3417/13/11/6833

[2] M. Rezaei and M. Jalili, "Knowledge Graph-Based Recommendation System Enhanced by Neural Collaborative Filtering and Knowledge Graph Embedding," Ain Shams Engineering Journal, vol. 15, no. 1, 2024. [Online]. Tersedia: https://www.sciencedirect.com/science/article/pii/S2090447923001521

[3] P. A. S. Mukti and Z. K. A. Baizal, "Enhancing Neural Collaborative Filtering with Metadata for Book Recommender System," Indonesian Journal of Computing and Cybernetics Systems, vol. 19, no. 1, 2025. [Online]. Tersedia: https://journal.ugm.ac.id/ijccs/article/view/103611
