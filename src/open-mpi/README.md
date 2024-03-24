# Tugas Kecil - Paralel Inverse Matrix
## K02 - Kelompok 13 - Siskom

Paralelisasi program dilakukan pada fungsi `performMatrixOperations` dengan menggunakan `MPI_Bcast` untuk membagikan informasi terkait besar matriks ke seluruh node / proses lain, diikuti dengan `MPI_Scatterv` untuk membagi matriks menjadi chunks berdasarkan jumlah node yang tersedia. Setelah itu, setiap node akan memanggil fungsi `matrix_inversion` untuk melakukan eliminasi Gauss-Jordan secara independen di setiap chunk matriksnya masing-masing. Pada fungsi tersebut juga terdapat pembuatan matriks identitas oleh masing-masing node pada chunk matriksnya. Jika seluruh perhitungan telah selesai, hasilnya akan dikirimkan dan disatukan kembali menjadi suatu matriks dengan menggunakan fungsi `MPI_Gatherv`.

Cara program membagikan data:
`MPI_Comm_rank` digunakan untuk mengetahui rank dari sebuah proses / node dan `MPI_Comm_size` untuk mengetahui banyak node yang akan digunakan. Informasi tersebut kemudian digunakan oleh `MPI_Scatterv` untuk 
membagi matriks menjadi chunks berdasarkan jumlah node yang tersedia.
Setiap node akan memegang banyak row yang sama, namun jika tidak bisa dibagi rata maka sisanya akan dibagikan ke node dengan rank terkecil terlebih dahulu.

Alasan pemilihan skema pembagian:
`MPI_Scatterv`, `MPI_Gatherv`, dan `MPI_Bcast` dapat membagi dan menggabungkan matriks secara efisien karena mergurus detail dari komunikasi antar proses secara kolektif sehingga mengurangi overhead terkait komunikasi. Skema ini juga memastikan bahwa setiap proses mendapatkan bagian yang seimbang dari data sehingga tidak akan ada proses yang bebannya berlebih. 

`MPI_Scatterv` dan `MPI_Gatherv` dipilih ketimbang `MPI_Scatter` dan `MPI_Gather` karena menggunakan vektor sehingga tetap dapat membagi baris bahkan jika tidak bisa dibagi rata dengan jumlah node / proses.

`MPI_Bcast` dan `MPI_Gatherv` dipilih ketimbang `MPI_Send` dan `MPI_Recv` karena bersifat kolektif sehingga menambah efektivitas dan mengurangi overhead komunikasi.


## How to Run
### Contoh build, run, dan menyimpan output untuk test case `32.txt`.
Serial
`time ./serial < 32.txt > output.txt`

```console
user@user:~/kit-tucil-sister-2024$ make
user@user:~/kit-tucil-sister-2024$ cat ./test_cases/32.txt | ./bin/serial > 32.txt
```

Paralel (lokal)
compile: `mpicc mpi.c -o mpi`
run: `time mpirun -np 4 ./mpi ./32.txt > output.txt`

Paralel (server)
`time mpirun -np 4 --hostfile hostfile.txt ./mpi 32.txt > out.txt`

