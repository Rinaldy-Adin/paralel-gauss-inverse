# Tugas Kecil - Paralel Inverse Matrix
## K02 - Kelompok 13 - Siskom

Paralelisasi perhitungan invers matrix dengan metode Gauss Jordan terjadi dalam fungsi `matrixInvesion` yang akan jalan pada setiap thread GPU yang teralokasikan pada program. Program sendiri pada awalnya akan mengalokasikan 1 blok thread, berjumlah sebanyak row yang ada pada matrix input. Setiap thread akan menghitung invers pada satu row, atau lebih jika jumlah row yang ada melebihi maksimal jumlah thread pada satu blok (1024 thread). Setiap thread dalam program akan melakukan iterasi untuk setiap row, dengan thread yang bertanggung jawab atas pivot row melakukan normalisasi (agar matrix menjadi matrix diagonal) pada row pivot, sedangkan thread yang lain akan menunggu proses normalisasi tersebut pada panggilan `__syncthreads()`. Setelah terjadi normalisasi pada row pivot, setiap row, kecuali row pivot, akan mengurangi row tersebut dengan row pivot yang dikali dengan faktor (elemen row pada kolom yang sama dengan elemen diagonal bernilai 1 pada row pivot) untuk row tersebut. Setelah semua iterasi selesai, hasil dari invers akan ada pada letak matrix yang awalnya dialokasikan sebagai matrix identitas. Program kami mengatasi edge case dimana jumlah row yang ada melebihi jumlah thread maksimal pada suatu blok, masalah tersebut diatasi dengan mengalokasikan lebih dari satu row pada beberapa thread sehingga thread tersebut bertanggung jawab untuk mengkalkulasi beberapa row yang berbeda.

Cara program membagikan data:
Program CUDA kami membagikan data dengan memanfaatkan 1 blok saja dengan sebanyak mungkin thread. Satu thread minimal memegang satu row. Data matrix yang dibagikan akan pertama dialokasikan terlebih dahulu kedalam global memory GPU, lalu setiap thread akan mengetahui isi sub-matrix yang akan thread tersebut pegang dari nomor thread yang didapatkan dari `threadIdx.x`. Selain itu, untuk mempercepat pembacaan pivot row yang pasti dilakukan oleh setiap thread, maka pivot row akan disimpan pada shared memory dari blok yang digunakan.

Alasan pemilihan skema pembagian:
Program kami hanya memanfaatkan 1 blok GPU saja karena urutan alogritma Gauss Jordan yang bersifat sekuensial bagi keseluruhan matrix. Alogritma Gauss Jordan sekuensial karena setiap iterasi bergantung pada hasil keseluruhan matrix pada iterasi sebelumnya. Hal tersebut membatasi pemanfaatan blok GPU karena tidak bisa dilakukan orkestrasi/sinkronisasi antar thread pada blok GPU yang berbeda sehingga hanya digunakan 1 blok GPU agar sinkronisasi antar semua thread di program tetap dapat dilakukan. Pada setiap iterasi juga memanfaatkan shared memory untuk membagikan pivot row, hal tersebut dilakukan karena pivot row merupakan kumpulan data yang sama yang diakses sangat sering pada suatu iterasi sehingga program dapat mendapatkan efisiensi lebih ketika akses memori tersebut dipercepat dengan memanfaatkan shared memory, yang dapat lebih cepat diakses oleh suatu thread daripada akses ke global memory.

## How to Run
### Contoh build, run, dan menyimpan output untuk test case `32.txt`.
Serial
`time ./serial < 32.txt > output.txt`

```console
user@user:~/kit-tucil-sister-2024$ make
user@user:~/kit-tucil-sister-2024$ cat ./test_cases/32.txt | ./bin/serial > 32.txt
```

Paralel
compile: `nvcc cuda.cu -o ./cuda`
run: `ime ./cuda < ./32.txt > ./output.txt`