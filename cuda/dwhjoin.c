#include <stdio.h>
#include <stdlib.h>

#define KNZ_LEN 20
#define DIM_COUNT 5
#define DIM_SIZE 1000
#define FACT_SIZE 100000

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

__device__ int nvstrncmp(char *a, char *b, int len) {
    for(int i = 0; i < len; i++) {
        if(a[i] == 0 && b[i] == 0) {
            return 0;
        }
        if(a[i] == 0 || b[i] == 0) {
            return 1;
        }
        if(a[i] != b[i]) {
            return 1;
        }
    }
    return 0;
}

__global__ void join(DimTable *dim, int dim_len, 
          FactTableIn *in, FactTableOut *out, int fact_len, 
          int idx) 
{
    int t = threadIdx.x;
    int max = blockDim.x;
    int len = (fact_len / max);
    int start = len * t;
    for(int i = start; i < len; i++) {        
        for(int j = 0; j < dim_len; j++) {
            if(nvstrncmp(dim[j].knz, in[i].knz[idx], KNZ_LEN) == 0) {
                out[i].id[idx] = dim[j].id;
                j = dim_len; // den inneren Schleifendurchlauf vorzeitig abbrechen
            }
        }
    }
    out[0].id[0] = 100;
}

int main(int argc, char *argv[]) {
    // Dimensionsdaten vorbereiten (jede genau DIM_SIZE Einträge) 
    //                             (das ist zwar unrealistisch hier aber praktisch)
    DimTable **dims = (DimTable**)calloc(sizeof(DimTable*), DIM_COUNT);
    for(int i = 0; i < DIM_COUNT; i++) {
        dims[i] = (DimTable*)calloc(sizeof(DimTable), DIM_SIZE);
        for(int j = 0; j < DIM_SIZE; j++) {
            dims[i][j].id = j;
            sprintf(dims[i][j].knz, "KNZ%d-%d", i, j);
        }
    }

    // Faktendaten vorbereiten
    FactTableIn *factIn = (FactTableIn*)calloc(sizeof(FactTableIn), FACT_SIZE);
    FactTableOut *factOut = (FactTableOut*)calloc(sizeof(FactTableOut), FACT_SIZE);
    for(int i = 0; i < DIM_COUNT; i++) {
        for(int j = 0; j < FACT_SIZE; j++) {
            sprintf(factIn[j].knz[i], "KNZ%d-%d", i % DIM_COUNT, j % DIM_SIZE);
        }
    }

    // Daten kopieren
    DimTable *dev_dim;
    FactTableIn *dev_factIn;
    FactTableOut *dev_factOut;

    cudaMalloc(&dev_dim, sizeof(DimTable) * DIM_SIZE);
    cudaMalloc(&dev_factIn, sizeof(FactTableIn) * FACT_SIZE);
    cudaMalloc(&dev_factOut, sizeof(FactTableOut) * FACT_SIZE);

    cudaMemcpy(dev_factIn, factIn, sizeof(FactTableIn) * FACT_SIZE, cudaMemcpyHostToDevice);

    // Join berechnen
    printf("Fakten und Dimmensionen vorbereitet\n");
    for(int i = 0; i < DIM_COUNT; i++) {
        cudaMemcpy(dev_dim, dims[i], sizeof(DimTable) * DIM_SIZE, cudaMemcpyHostToDevice);
        join<<<1, 100>>>(dev_dim, DIM_SIZE, 
             dev_factIn, dev_factOut, FACT_SIZE, i);
        cudaDeviceSynchronize();
        cudaFree(dev_dim);
    }    
    printf("Join abgeschlossen\n");

    // Daten zurückkopieren
    cudaMemcpy(factOut, dev_factOut, sizeof(FactTableOut) * FACT_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(dev_factIn);
    cudaFree(dev_factOut);

    // Ergebnis ausgeben
    printf("Top 5:\n");
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < DIM_COUNT; j++) {
            printf("%s->%ld | ", factIn[i].knz[j], factOut[i].id[j]);
        }
        printf("\n");
    }
    
    free(factIn);
    for(int i = 0; i < DIM_COUNT; i++) {
        free(dims[i]);
    }
    free(dims);
    free(factOut);

    return 0;
}