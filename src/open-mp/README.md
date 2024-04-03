# Tugas Kecil 2 - Paralel Inverse Matrix
## K02 - Kelompok 13 - Siskom

Bagian utama dari paralelisasi program terjadi dalam fungsi `matrixInversion` dan `partialPivoting`, di mana OpenMP membantu mendistribusikan beban kerja di antara beberapa thread untuk mempercepat proses eliminasi Gauss-Jordan dan pemilihan pivot untuk menemukan elemen pivot maksimum.

Pada tugas kecil ini, juga disediakan kode serupa yang masih serial untuk membandingkan kinerja antara kedua kode dengan lebih *apple to apple*. 

- **Partial Pivoting**: Fungsi `partialPivoting` menggunakan OpenMP untuk mencari baris dengan elemen pivot maksimum  paralel, yang kemudian dipertukarkan dengan baris pivot jika diperlukan.

- **Eliminasi Gauss-Jordan**: Bagian eliminasi dalam `matrixInversion` diparalelisasi, memungkinkan setiap thread untuk bekerja pada barisnya sendiri secara bersamaan. Hal ini mengurangi waktu yang diperlukan untuk mengubah matriks menjadi invers.

## How to Run
### Contoh build, run, dan menyimpan output untuk test case `32.txt`.
Serial
`time ./serial < 32.txt > output.txt`

Paralel
compile: `g++ -fopenmp mp.cpp -o mp`
run: `time ./mp < 32.txt > output.txt`