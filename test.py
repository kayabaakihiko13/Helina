angka1 = int(input("masukkan angka:"))
angka2 = int(input("masukkan angka: "))

try:
    print(angka1 / angka2)
except ZeroDivisionError:
    print("tidak bisa dibagi dengan 0")
