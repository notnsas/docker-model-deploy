# Prediksi Model (Docker, Flask, & Tailwind)

Proyek ini adalah aplikasi web sederhana untuk mendemonstrasikan deployment model machine learning. Aplikasi ini memiliki dua mode: prediksi realtime (melalui form) dan prediksi batch (melalui upload file CSV/Excel).

Aplikasi ini dibangun menggunakan **Flask** untuk backend, **Tailwind CSS** untuk frontend, dan di-deploy dalam container **Docker**.

**(Di sini tempat kamu tambahkan gambar/screenshot utama aplikasi kamu)**

## Fitur

  * **Prediksi Realtime:** Masukkan data via form dan dapatkan hasil prediksi secara instan.
  * **Prediksi Batch:** Upload satu atau beberapa file (CSV/Excel) untuk mendapatkan prediksi massal.
  * **Penggabungan Data (Join):** Mendukung upload beberapa file dan menggabungkannya (join) berdasarkan kolom kunci, mirip seperti `JOIN` pada SQL.
  * **Validasi Data:** Menggunakan Pydantic untuk validasi data input.
  * **Deployment Mudah:** Siap digunakan dengan Docker.

## Teknologi

  * **Backend:** Flask
  * **Frontend:** Tailwind CSS
  * **Validasi Data:** Pydantic
  * **Deployment:** Docker

-----

## Cara Menjalankan Aplikasi

Ada dua cara untuk menjalankan aplikasi ini:

### Opsi 1: Menjalankan dengan Docker (Disarankan)

Cara ini adalah yang paling mudah dan cepat, karena semua dependensi sudah dikemas dalam image Docker.

1.  Pastikan kamu sudah menginstal dan menjalankan **Docker Desktop** (untuk Windows/Mac).

2.  Buka terminal atau command prompt.

3.  (Opsional) Login ke Docker Hub jika diperlukan:

    ```bash
    docker login
    ```

4.  Tarik (pull) image dari Docker Hub:

    ```bash
    docker pull notnsas/model-deployment
    ```

5.  Jalankan container dari image tersebut:

    ```bash
    docker run -d -p 8080:5000 notnsas/model-deployment
    ```

      * `-d` menjalankan container di background (detached mode).
      * `-p 8080:5000` memetakan port 8080 di komputermu ke port 5000 di dalam container (karena Flask berjalan di port 5000).

6.  Selesai\! Buka browser dan akses aplikasi di: `http://localhost:8080`

### Opsi 2: Menjalankan Secara Lokal (Manual)

Cara ini digunakan jika kamu ingin menjalankan aplikasi langsung dari source code, misalnya untuk development.

1.  Clone repository ini atau unduh sebagai file ZIP.
    ```bash
    git clone https://github.com/notnsas/docker-model-deploy.git
    ```
2.  Masuk ke direktori proyek:
    ```bash
    cd docker-model-deploy
    ```
3.  Buat *virtual environment* (venv) untuk proyek ini:
    ```bash
    python -m venv venv
    ```
4.  Aktifkan venv:
      * **Di Windows (CMD/PowerShell):**
        ```bash
        venv\Scripts\activate
        ```
      * **Di macOS/Linux (Bash):**
        ```bash
        source venv/bin/activate
        ```
5.  Install semua library yang dibutuhkan dari file `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
6.  Jalankan aplikasi Flask:
    ```bash
    flask run
    ```
7.  Selesai\! Buka browser dan akses aplikasi di: `http://localhost:5000`

-----

## Cara Penggunaan Aplikasi

### Mode Realtime

**(Tambahkan screenshot untuk mode realtime di sini)**

1.  Buka aplikasi di browser.
2.  Scroll ke bagian Realtime di bawah.
3.  Masukkan data yang diminta pada form (Contoh: Amount dan Location).
4.  Klik tombol "Predict" untuk melihat hasil prediksi.

### Mode Batch (Upload File)

Mode ini digunakan untuk memproses banyak data sekaligus dari file.

**(Tambahkan screenshot untuk mode batch/upload di sini)**

1.  Scroll bagian Upload yaitu diatas.
2.  Klik **"Upload First File"** untuk meng-upload file data pertamamu (CSV/Excel).
3.  **(Opsional) Jika data terpisah di file lain:**
      * Klik **"Add Another File"** untuk menambahkan file kedua (atau ketiga, dst.). Ini berguna jika kamu memiliki data yang perlu digabung, misalnya data transaksi di satu file dan data customer di file lain.
4.  **(Opsional) Konfigurasi Join:**
      * Jika kamu meng-upload lebih dari satu file, kamu perlu mengatur konfigurasinya di bagian **"Join Configuration"**.
      * Pilih "Table" (file) mana yang ingin kamu gabungkan.
      * Tentukan **kolom kunci** (common column) dari kedua table yang akan digunakan sebagai dasar join (Contoh: `customer_id` di file A dan `customer_id` di file B).
      * Klik **"Set Join Columns"** untuk mengkonfirmasi.
5.  Kamu bisa menambah table lain dengan **"Add New Table"** atau menghapus dengan **"Delete Table"**.
6.  **Dapatkan Prediksi:**
      * Setelah semua file di-upload dan konfigurasi join (jika ada) selesai, klik tombol proses/upload (misalnya "Get Prediction" atau "Upload").
      * Aplikasi akan memproses data dan memberikanmu link untuk mengunduh hasil prediksi dalam format CSV/Excel.

### Contoh Data untuk Mode Batch

Kamu bisa mencoba fungsionalitas upload batch menggunakan contoh dataset di link Google Drive berikut:
[Link Contoh Dataset](https://drive.google.com/drive/folders/1YnBWbqUJUVb2IXumbV7cY0S4B0JcZKZ5?usp=sharing)

**Skenario Uji Coba:**

1.  Upload `cust_transaction_details (1).csv` sebagai file pertama.
2.  Klik "Add Another File" dan upload `Customer_DF (1).csv` sebagai file kedua.
3.  Di "Join Configuration", atur join antara kedua table tersebut menggunakan kolom kuncinya (misalnya, kolom ID customer yang sama di kedua file).
4.  Klik "Set Join Columns" lalu proses untuk mendapatkan prediksi.

-----

## Penjelasan Struktur Kode (Sederhana)

  * `app/routes.py`: Ini adalah file utama backend Flask. File ini yang:
      * Me-load model machine learning saat aplikasi dijalankan.
      * Menerima request (permintaan) dari frontend (UI).
      * Membedakan apakah ini request 'Realtime' (dari form) atau 'Batch' (dari upload file).
      * Memanggil fungsi prediksi yang sesuai.
  * `utils/`: Folder ini berisi script Python tambahan untuk:
      * **Preprocessing:** Membersihkan dan menyiapkan data sebelum dimasukkan ke model.
      * **Validasi:** Menggunakan Pydantic untuk memastikan data yang masuk (baik dari form atau file) memiliki format yang benar.
  * `templates/` & `static/`: Berisi file HTML dan CSS (Tailwind) untuk tampilan antarmuka (UI/UX).
  * **Alur Batch:** Saat user meng-upload beberapa file, backend akan:
    1.  Menerima file-file tersebut.
    2.  Membacanya sebagai dataframe.
    3.  Menggabungkannya (join) sesuai konfigurasi user.
    4.  Melakukan preprocessing dan validasi.
    5.  Menjalankan prediksi model pada data yang sudah bersih.
    6.  Membuat file CSV/Excel baru berisi hasil prediksi dan mengirimkannya kembali ke user.