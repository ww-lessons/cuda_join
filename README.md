Mit den folgenden Parametern hab ich auf meinem alten Core2Duo mit einer NVidia Geforce GT 710 mit 2 GB RAM 
die folgenden Werte erhalten.

PARAMETER:

#define KNZ_LEN 20
#define DIM_COUNT 3
#define DIM_SIZE 10000
#define FACT_SIZE 250000

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

real    0m44.232s
user    0m44.196s
sys     0m0.004s
wiw39784@rechner:~/Dokumente/c/dwhjoin_experiment/bin$ time ./dwhjoin_openmp
Fakten und Dimmensionen vorbereitet
Join abgeschlossen
Top 5:
KNZ0-0->0 | KNZ1-0->0 | KNZ2-0->0 |
KNZ0-1->1 | KNZ1-1->1 | KNZ2-1->1 |
KNZ0-2->2 | KNZ1-2->2 | KNZ2-2->2 |
KNZ0-3->3 | KNZ1-3->3 | KNZ2-3->3 |
KNZ0-4->4 | KNZ1-4->4 | KNZ2-4->4 |

real    0m28.967s
user    0m55.112s
sys     0m0.008s
wiw39784@rechner:~/Dokumente/c/dwhjoin_experiment/bin$ time ./dwhjoin_cuda (mit 100 CUDA-Cores)
Fakten und Dimmensionen vorbereitet
Join abgeschlossen
Top 5:
KNZ0-0->0 | KNZ1-0->0 | KNZ2-0->0 |
KNZ0-1->1 | KNZ1-1->1 | KNZ2-1->1 |
KNZ0-2->2 | KNZ1-2->2 | KNZ2-2->2 |
KNZ0-3->3 | KNZ1-3->3 | KNZ2-3->3 |
KNZ0-4->4 | KNZ1-4->4 | KNZ2-4->4 |

real    0m14.470s
user    0m10.580s
sys     0m3.620s

wiw39784@rechner:~/Dokumente/c/dwhjoin_experiment/bin$ time ./dwhjoin_cuda (mit 200 CUDA-Cores)
Fakten und Dimmensionen vorbereitet
Join abgeschlossen
Top 5:
KNZ0-0->0 | KNZ1-0->0 | KNZ2-0->0 |
KNZ0-1->1 | KNZ1-1->1 | KNZ2-1->1 |
KNZ0-2->2 | KNZ1-2->2 | KNZ2-2->2 |
KNZ0-3->3 | KNZ1-3->3 | KNZ2-3->3 |
KNZ0-4->4 | KNZ1-4->4 | KNZ2-4->4 |

real    0m3.993s
user    0m2.868s
sys     0m1.068s