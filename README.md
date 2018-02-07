Mit den folgenden Parametern hab ich auf meinem alten Core2Duo mit einer NVidia Geforce GT 710 mit 2 GB RAM 
die folgenden Werte erhalten.

PARAMETER:

#define KNZ_LEN 20
#define DIM_COUNT 3
#define DIM_SIZE 4000
#define FACT_SIZE 1000000

WERTE:

wiw39784@rechner:~/Dokumente/c/dwhjoin_experiment/bin$ time ./dwhjoin_simple
Fakten und Dimmensionen vorbereitet
Join abgeschlossen
Top 5:
KNZ0-0->0 | KNZ1-0->0 | KNZ2-0->0 |
KNZ0-1->1 | KNZ1-1->1 | KNZ2-1->1 |
KNZ0-2->2 | KNZ1-2->2 | KNZ2-2->2 |
KNZ0-3->3 | KNZ1-3->3 | KNZ2-3->3 |
KNZ0-4->4 | KNZ1-4->4 | KNZ2-4->4 |

real    1m8.665s
user    1m8.400s
sys     0m0.084s
wiw39784@rechner:~/Dokumente/c/dwhjoin_experiment/bin$ time ./dwhjoin_openmp
Fakten und Dimmensionen vorbereitet
Join abgeschlossen
Top 5:
KNZ0-0->0 | KNZ1-0->0 | KNZ2-0->0 |
KNZ0-1->1 | KNZ1-1->1 | KNZ2-1->1 |
KNZ0-2->2 | KNZ1-2->2 | KNZ2-2->2 |
KNZ0-3->3 | KNZ1-3->3 | KNZ2-3->3 |
KNZ0-4->4 | KNZ1-4->4 | KNZ2-4->4 |

real    0m44.318s
user    1m25.012s
sys     0m0.084s
wiw39784@rechner:~/Dokumente/c/dwhjoin_experiment/bin$ time ./dwhjoin_cuda
Fakten und Dimmensionen vorbereitet
Join abgeschlossen
Top 5:
KNZ0-0->0 | KNZ1-0->0 | KNZ2-0->0 |
KNZ0-1->0 | KNZ1-1->0 | KNZ2-1->0 |
KNZ0-2->0 | KNZ1-2->0 | KNZ2-2->0 |
KNZ0-3->0 | KNZ1-3->0 | KNZ2-3->0 |
KNZ0-4->0 | KNZ1-4->0 | KNZ2-4->0 |

real    0m11.300s
user    0m7.892s
sys     0m3.356s