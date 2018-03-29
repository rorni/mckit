#include <string.h>
#include "coverage.h"

void coverages(
    char * arr,
    size_t nrow,
    size_t ncol,
    char value
)
{
    char * urows = (char *) malloc(nrow * sizeof(char));
    char * ucols = (char *) malloc(ncol * sizeof(char));
    memset(urows, 0, nrow);
    memset(ucols, 0, ncol);
    
    find_coverage(arr, nrow, ncol, value, urows, ucols);
}

void find_coverage(
    char * arr,           // Initial array
    size_t nrow,          // Number of rows
    size_t ncol,          // Number of columns
    char value,           // Value for which the coverage is searched.
    const char * urows,   // Used rows - 1 - the row is already used.
    const char * ucols    // Used columns - 1 - the column is already used.
)
{
    char * frows = malloc(nrow * sizeof(char));
    memcpy(frows, urows, nrow * sizeof(char));
    
    int j, i = row_min_index(arr, nrow, ncol, value, urows, ucols);
    if (i < 0) return // It is impossible to find coverage;
    
    size_t rows_left;
    for (j = 0; j < ncol; ++j) {
        if (ucols[j] != 0) continue;
        if (*(arr + i * ncol + j) == value) {
            ucols[j] = 1;
            rows_left = mark_used_rows(arr, nrow, ncol, value, frows, j);
            if (rows_left == 0) {
                // return this number
            } else {
                // get variants
                find_coverage(arr, nrow, ncol, value, frows, ucols);
            }
            ucols[j] = 0;
            memcpy(frows, urows, nrow * sizeof(char));
        }
    }    
    free(frows);
}

size_t mark_used_rows(
    char * arr, 
    size_t nrow,
    size_t ncol,
    char value,
    char * urows,
    size_t mcol
)
{
    size_t i, j, cnt;
    for (i = 0; i < nrow; ++i) {
        if (urows[i] == 0) {
            if (*(arr + i * ncol + mcol) == value) urows[i] = 1;
            else ++cnt;
        }         
    }
    return cnt;
}

int row_min_index(
    char * arr, 
    size_t nrow, 
    size_t ncol, 
    char value, 
    const char * urows, 
    const char * ucols
) 
{
    size_t min_i = -1, i, j;
    int min_cnt = SIZE_MAX, cnt;
    for (i = 0; i < nrow; ++i) {
        if (urows[i] != 0) continue;
        cnt = 0;
        for (j = 0; j < ncol; ++j) {
            if (ucols[j] == 0 && *(arr + i * ncol + j) == value) ++cnt;
        }
        if (cnt > 0 && cnt < min_cnt) {
            min_cnt = cnt;
            min_i = i;
        }
    }
    return min_i;
}