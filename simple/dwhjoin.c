#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KNZ_LEN 20
#define DIM_COUNT 3
#define DIM_SIZE 10000
#define FACT_SIZE 250000

// Datenstruktur

typedef struct _dim {
    long id;
    char knz[KNZ_LEN];
} DimTable;

typedef struct _factIn {
    char knz[DIM_COUNT][KNZ_LEN];
} FactTableIn;

typedef struct _factOut {
    long id[DIM_COUNT];
} FactTableOut;

// Funktion

void join(DimTable *dim, int dim_len, 
          FactTableIn *in, FactTableOut *out, int fact_len, 
          int idx) 
{
    for(int i = 0; i < fact_len; i++) {
        char *s = in[i].knz[idx];
        for(int j = 0; j < dim_len; j++) {
            if(strncmp(dim[j].knz, s, KNZ_LEN) == 0) {
                out[i].id[idx] = dim[j].id;
                j = dim_len; // den inneren Schleifendurchlauf vorzeitig abbrechen
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Dimensionsdaten vorbereiten (jede genau DIM_SIZE Einträge) 
    //                             (das ist zwar unrealistisch hier aber praktisch)
    DimTable **dims = calloc(sizeof(DimTable*), DIM_COUNT);
    for(int i = 0; i < DIM_COUNT; i++) {
        dims[i] = calloc(sizeof(DimTable), DIM_SIZE);
        for(int j = 0; j < DIM_SIZE; j++) {
            dims[i][j].id = j;
            sprintf(dims[i][j].knz, "KNZ%d-%d", i, j);
        }
    }

    // Faktendaten vorbereiten
    FactTableIn *factIn = calloc(sizeof(FactTableIn), FACT_SIZE);
    FactTableOut *factOut = calloc(sizeof(FactTableOut), FACT_SIZE);
    for(int i = 0; i < DIM_COUNT; i++) {
        for(int j = 0; j < FACT_SIZE; j++) {
            sprintf(factIn[j].knz[i], "KNZ%d-%d", i % DIM_COUNT, j % DIM_SIZE);
        }
    }

    // Join berechnen
    printf("Fakten und Dimmensionen vorbereitet\n");
    for(int i = 0; i < DIM_COUNT; i++) {
        join(dims[i], DIM_SIZE, 
             factIn, factOut, FACT_SIZE, i);
    }
    printf("Join abgeschlossen\n");

    // Ergebnis ausgeben
    printf("Top 5:\n");
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < DIM_COUNT; j++) {
            printf("%s->%ld | ", factIn[i].knz[j], factOut[i].id[j]);
        }
        printf("\n");
    }

    // Speicher freigeben
    for(int i = 0; i < DIM_COUNT; i++) {
        free(dims[i]);
    }
    free(dims);
    free(factIn);
    free(factOut);

    return 0;
}