# Tugas Kecil - Paralel Inverse Matrix

Paralelisasi prograrm dilakukan dengan menggunakan MPI_Comm_rank dan MPI_Comm_size untuk mendapatkan rank dari masing-masing proses dan jumlah total proses yang sedang berjalan. Untuk mencegah duplikasi maka matriks hanya akan dibaca pada rank 0, lalu kemudian ukurannya akan dibroadcast dengan menggunakan MPI_Bcast dan input dibagi antara proses-proses dengan menggunakan MPI_Scatter. Setelah itu, proses inversi dilakukan pada matriks lokal dan hasilnya dikumpulkan kembali di rank 0 dengan menggunakan MPI_Gather. Hasil dari setiap proses tersebut kemudian akan digabungkan lagi sehingga akhirnya menjadi matriks yang lengkap.

Alasan pemilihan skema pembagian:
MPI_SCatter dan MPI_Gather dapat membagi dan menggabungkan matriks secara efisien karena mergurus detail dari komunikasi antar proses sehingga mengurangi overhead terkait komunikasi. Skema ini juga memastikan bahwa setiap proses mendapatkan bagian yang seimbang dari data sehingga tidak akan ada proses yang bebannya berlebih.


## How to Run
### Contoh build, run, dan menyimpan output untuk test case `32.txt`.
Serial
`time ./serial < 32.txt > output.txt`

Paralel
`mpicc mpi.c -o mpi_program`
`time mpirun -np 2 ./mpi_program ./32.txt > output.txt`

```console
user@user:~/kit-tucil-sister-2024$ make
user@user:~/kit-tucil-sister-2024$ cat ./test_cases/32.txt | ./bin/serial > 32.txt
```
