# Current

## Bit

| Scheme                 | Parameters                                       | Slots |     Add |      Mul | Bootstrap | Mul/Bootstrap | Amortized Add | Amortized Mul |
| :--------------------- | :----------------------------------------------- | ----: | ------: | -------: | --------: | ------------: | ------------: | ------------: |
| Cyclotomic-Subring BFV | n = 32768, m = 3342387, log2(α) = 850, t = 2^1   |  4096 |  5.0 ms | 390.0 ms |   24.36 s |            17 |        1.2 µs |      445.1 µs |
| Cyclotomic-Subring BFV | n = 32768, m = 4259905, log2(α) = 850, t = 2^1   |  8192 |  5.0 ms | 391.0 ms |   33.01 s |            18 |        610 ns |      271.6 µs |
| Cyclotomic-Subring BFV | n = 49152, m = 10158235, log2(α) = 1280, t = 2^1 | 12288 | 14.0 ms |   1.03 s |   1.1 min |            31 |        1.1 µs |      249.0 µs |
| Cyclotomic-Subring BFV | n = 49152, m = 35297665, log2(α) = 1280, t = 2^1 | 12288 | 14.0 ms | 982.0 ms |   45.79 s |            22 |        1.1 µs |      249.3 µs |
| Cyclotomic BFV         | n = 31104, log2(q) = 812, t = 2                  |   864 |  1.0 ms |   1.19 s |   27.65 s |            34 |        1.2 µs |        2.3 ms |
| Cyclotomic BFV         | n = 42336, log2(q) = 1110, t = 2                 |  2016 |  1.0 ms |   1.64 s |   43.27 s |            52 |        496 ns |        1.2 ms |

## Uint64

| Scheme                  | Parameters                                                 | Slots |      Add |     Mul | Bootstrap | Mul/Bootstrap | Amortized Add | Amortized Mul |
| :---------------------- | :--------------------------------------------------------- | ----: | -------: | ------: | --------: | ------------: | ------------: | ------------: |
| Cyclotomic-Subring DBFV | n = 32768, m = 3342387, log2(α) = 850, t = 2^64, b = 2^6   |  4096 |  92.0 ms |  5.32 s |  14.0 min |             5 |       22.5 µs |       42.3 ms |
| Cyclotomic-Subring DBFV | n = 32768, m = 3342387, log2(α) = 850, t = 2^64, b = 2^8   |  4096 |  73.0 ms |  3.83 s |   8.6 min |             3 |       17.8 µs |       43.1 ms |
| Cyclotomic-Subring DBFV | n = 32768, m = 4259905, log2(α) = 850, t = 2^64, b = 2^6   |  8192 |  93.0 ms |  5.40 s |  23.0 min |             5 |       11.4 µs |       34.3 ms |
| Cyclotomic-Subring DBFV | n = 32768, m = 4259905, log2(α) = 850, t = 2^64, b = 2^8   |  8192 |  69.0 ms |  3.92 s |  13.6 min |             3 |        8.4 µs |       33.6 ms |
| Cyclotomic-Subring DBFV | n = 49152, m = 10158235, log2(α) = 1280, t = 2^64, b = 2^6 | 12288 | 291.0 ms | 13.35 s |  31.0 min |            17 |       23.7 µs |       10.0 ms |
| Cyclotomic-Subring DBFV | n = 49152, m = 10158235, log2(α) = 1280, t = 2^64, b = 2^8 | 12288 | 197.0 ms |  9.38 s |  22.0 min |            13 |       16.0 µs |        9.0 ms |
| Cyclotomic-Subring DBFV | n = 49152, m = 35297665, log2(α) = 1280, t = 2^64, b = 2^6 | 12288 | 287.0 ms | 13.17 s |  22.1 min |            12 |       23.4 µs |       10.1 ms |
| Cyclotomic-Subring DBFV | n = 49152, m = 35297665, log2(α) = 1280, t = 2^64, b = 2^8 | 12288 | 196.0 ms |  9.01 s |  16.2 min |            10 |       16.0 µs |        8.7 ms |
| TFHE                    | n = 918, N = 2048                                          |     1 | 247.0 ms |  3.98 s |         — |             ∞ |      247.0 ms |        3.98 s |

## Linear Transform (Coefficients-to-Slots)

| Scheme                 | Parameters                                       | Type | Slots |    Time | Amortized | Noise Growth |
| :--------------------- | :----------------------------------------------- | ---: | ----: | ------: | --------: | -----------: |
| Cyclotomic-Subring BFV | n = 32768, m = 3342387, log2(α) = 850, t = 2^1   |  Bit |  4096 | 12.29 s |    3.0 ms |      19 bits |
| Cyclotomic-Subring BFV | n = 32768, m = 4259905, log2(α) = 850, t = 2^1   |  Bit |  8192 | 19.78 s |    2.4 ms |      19 bits |
| Cyclotomic-Subring BFV | n = 49152, m = 10158235, log2(α) = 1280, t = 2^1 |  Bit | 12288 | 26.38 s |    2.1 ms |      19 bits |
| Cyclotomic-Subring BFV | n = 49152, m = 35297665, log2(α) = 1280, t = 2^1 |  Bit | 12288 | 11.82 s |  961.6 µs |      24 bits |
| Cyclotomic BFV         | n = 31104, log2(q) = 812, t = 2                  |  Bit |   864 | 14.39 s |   16.7 ms |      22 bits |
| Cyclotomic BFV         | n = 42336, log2(q) = 1110, t = 2                 |  Bit |  2016 | 33.56 s |   16.6 ms |      20 bits |

# Old (incomplete, low-noise-growth fields, aggressive parameter choice)

## Bit

| Scheme                 | Parameters                         | Slots |     Add |      Mul |   Bootstrap | Mul/Bootstrap | Amortized Add | Amortized Mul |
| :--------------------- | :--------------------------------- | ----: | ------: | -------: | ----------: | ------------: | ------------: | ------------: |
| Cyclotomic-Subring BFV | n = 24576, log2(α) = 640, t = 2^1  |  2048 |  2.0 ms | 204.0 ms |      9.67 s |             9 |        977 ns |      624.3 µs |
| Cyclotomic-Subring BFV | n = 32768, log2(α) = 850, t = 2^1  |  8192 |  6.0 ms | 386.0 ms | 34.09 s (!) |            18 |        732 ns |      278.3 µs |
| Cyclotomic-Subring BFV | n = 49152, log2(α) = 1280, t = 2^1 | 12288 | 15.0 ms |   1.02 s |     48.46 s |            29 |        1.2 µs |      218.9 µs |
| Cyclotomic BFV         | n = 31104, log2(q) = 812, t = 2    |   864 |  1.0 ms |   1.20 s |     26.16 s |            34 |        1.2 µs |        2.3 ms |
| Cyclotomic BFV         | n = 42336, log2(q) = 1110, t = 2   |  2016 |  1.0 ms |   1.56 s |     41.70 s |            52 |        496 ns |        1.2 ms |

## Uint64

| Scheme                  | Parameters                                   | Slots |      Add |     Mul |    Bootstrap | Mul/Bootstrap | Amortized Add | Amortized Mul |
| :---------------------- | :------------------------------------------- | ----: | -------: | ------: | -----------: | ------------: | ------------: | ------------: |
| Cyclotomic-Subring DBFV | n = 24576, log2(α) = 640, t = 2^64, b = 2^6  |  2048 |  45.0 ms |  2.99 s |  4.7 min (!) |             0 |       22.0 µs |             — |
| Cyclotomic-Subring DBFV | n = 24576, log2(α) = 640, t = 2^64, b = 2^8  |  2048 |  34.0 ms |  2.05 s |  3.0 min (!) |             0 |       16.6 µs |             — |
| Cyclotomic-Subring DBFV | n = 32768, log2(α) = 850, t = 2^64, b = 2^6  |  8192 | 104.0 ms |  5.73 s |     21.7 min |             5 |       12.7 µs |       32.4 ms |
| Cyclotomic-Subring DBFV | n = 32768, log2(α) = 850, t = 2^64, b = 2^8  |  8192 |  54.0 ms |  3.69 s |     13.1 min |             3 |        6.6 µs |       32.5 ms |
| Cyclotomic-Subring DBFV | n = 49152, log2(α) = 1280, t = 2^64, b = 2^6 | 12288 | 297.0 ms | 12.95 s |     30.1 min |            18 |       24.2 µs |        9.2 ms |
| Cyclotomic-Subring DBFV | n = 49152, log2(α) = 1280, t = 2^64, b = 2^8 | 12288 | 207.0 ms |  9.32 s | 20.5 min (!) |            15 |       16.8 µs |        7.4 ms |
| TFHE                    | n = 918, N = 2048                            |     1 | 215.0 ms |  2.93 s |            — |             ∞ |      215.0 ms |        2.93 s |

## Linear Transform (Coefficients-to-Slots)

| Scheme                 | Parameters                         | Type | Slots |    Time | Amortized | Noise Growth |
| :--------------------- | :--------------------------------- | ---: | ----: | ------: | --------: | -----------: |
| Cyclotomic-Subring BFV | n = 24576, log2(α) = 640, t = 2^1  |  Bit |  2048 |  3.40 s |    1.7 ms |      24 bits |
| Cyclotomic-Subring BFV | n = 32768, log2(α) = 850, t = 2^1  |  Bit |  8192 | 21.48 s |    2.6 ms |      19 bits |
| Cyclotomic-Subring BFV | n = 49152, log2(α) = 1280, t = 2^1 |  Bit | 12288 | 28.89 s |    2.4 ms |      21 bits |
| Cyclotomic BFV         | n = 31104, log2(q) = 812, t = 2    |  Bit |   864 | 14.12 s |   16.3 ms |      22 bits |
| Cyclotomic BFV         | n = 42336, log2(q) = 1110, t = 2   |  Bit |  2016 | 32.63 s |   16.2 ms |      20 bits |

`(!)` = decryption failed, i.e. the noise overflowed; a noise budget given for such a measurement still refers to the resulting ciphertext, and may well be positive if the overflow happened during a lower-precision intermediate computation

# Very old (before low-noise-growth fields)

## Bit

| Scheme                 | Parameters                         | Slots |     Add |      Mul | Bootstrap | Mul/Bootstrap | Amortized Add | Amortized Mul |
| :--------------------- | :--------------------------------- | ----: | ------: | -------: | --------: | ------------: | ------------: | ------------: |
| Cyclotomic-Subring BFV | n = 32768, log2(α) = 850, t = 2^1  |  8192 |  6.0 ms | 376.0 ms |   13.12 s |             4 |        732 ns |      446.3 µs |
| Cyclotomic-Subring BFV | n = 36864, log2(α) = 960, t = 2^1  |  3072 |  7.0 ms | 514.0 ms |   18.52 s |            12 |        2.3 µs |      669.8 µs |
| Cyclotomic-Subring BFV | n = 49152, log2(α) = 1280, t = 2^1 | 12288 | 16.0 ms |   1.00 s |   45.58 s |            26 |        1.3 µs |      224.2 µs |
| Cyclotomic BFV         | n = 31104, log2(q) = 812, t = 2    |   864 |  1.0 ms |   1.18 s |   26.69 s |            34 |        1.2 µs |        2.3 ms |
| Cyclotomic BFV         | n = 42336, log2(q) = 1110, t = 2   |  2016 |  1.0 ms |   1.57 s |   43.00 s |            52 |        496 ns |        1.2 ms |

## Uint64

| Scheme                  | Parameters                                   | Slots |      Add |     Mul |   Bootstrap | Mul/Bootstrap | Amortized Add | Amortized Mul |
| :---------------------- | :------------------------------------------- | ----: | -------: | ------: | ----------: | ------------: | ------------: | ------------: |
| Cyclotomic-Subring DBFV | n = 32768, log2(α) = 850, t = 2^64, b = 2^6  |  8192 |  75.0 ms |  4.47 s | 5.8 min (!) |             ? |        9.2 µs |             — |
| Cyclotomic-Subring DBFV | n = 32768, log2(α) = 850, t = 2^64, b = 2^8  |  8192 |  56.0 ms |  3.27 s | 4.2 min (!) |             ? |        6.8 µs |             — |
| Cyclotomic-Subring DBFV | n = 36864, log2(α) = 960, t = 2^64, b = 2^6  |  3072 |  95.0 ms |  5.98 s |     8.4 min |             4 |       30.9 µs |       43.0 ms |
| Cyclotomic-Subring DBFV | n = 36864, log2(α) = 960, t = 2^64, b = 2^8  |  3072 |  71.0 ms |  4.46 s |     6.3 min |             2 |       23.1 µs |       63.2 ms |
| Cyclotomic-Subring DBFV | n = 49152, log2(α) = 1280, t = 2^64, b = 2^6 | 12288 | 172.0 ms | 11.49 s |    20.1 min |            14 |       14.0 µs |        7.9 ms |
| Cyclotomic-Subring DBFV | n = 49152, log2(α) = 1280, t = 2^64, b = 2^8 | 12288 | 126.0 ms |  8.34 s |    15.3 min |            12 |       10.3 µs |        6.9 ms |
| TFHE                    | n = 918, N = 2048                            |     1 | 219.0 ms |  3.00 s |           — |             ∞ |      219.0 ms |        3.00 s |

## Linear Transform (Coefficients-to-Slots)

| Scheme                 | Parameters                         | Type | Slots |    Time | Amortized | Noise Growth |
| :--------------------- | :--------------------------------- | ---: | ----: | ------: | --------: | -----------: |
| Cyclotomic-Subring BFV | n = 32768, log2(α) = 850, t = 2^1  |  Bit |  8192 |  2.06 s |  251.1 µs |      22 bits |
| Cyclotomic-Subring BFV | n = 36864, log2(α) = 960, t = 2^1  |  Bit |  3072 |  3.13 s |    1.0 ms |      25 bits |
| Cyclotomic-Subring BFV | n = 49152, log2(α) = 1280, t = 2^1 |  Bit | 12288 | 11.91 s |  969.0 µs |      24 bits |
| Cyclotomic BFV         | n = 31104, log2(q) = 812, t = 2    |  Bit |   864 | 14.08 s |   16.3 ms |      22 bits |
| Cyclotomic BFV         | n = 42336, log2(q) = 1110, t = 2   |  Bit |  2016 | 32.52 s |   16.1 ms |      20 bits |