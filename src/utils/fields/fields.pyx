
from utils.prints import logMsg

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

cdef inline float readFloatField(float* f, int m, int x, int y):
    return f[y * m + x];

cdef inline void writeFloatField(float* f, int m, int x, int y, float value):
    f[y * m + x] = value;

cdef float* initFloatField(int m, int n, float v):
    cdef float* result = <float *> malloc(m * n * sizeof(float));
    
    for i in range(n * m):
        result[i] = v
    
    return result;

cdef void mirrorFloatField(float* f, int m, int n):
    cdef int x, y;
    cdef float tmp;
    for y in range(n):
        for x in range(m / 2):
            tmp = readFloatField(f, m, x, y)
            writeFloatField(f, m, x, y, readFloatField(f, m, m-x-1, y))
            writeFloatField(f, m, m-1-x, y, tmp)

cdef void rotateFloatField(float* f, int m, int n):
    cdef float* tmp = initFloatField(m, n, 0);
    memcpy(tmp, f, m * n * sizeof(float));
    
    cdef int x, y
    for y in range(n):
        for x in range(m):
            writeFloatField(f, m, x, y, readFloatField(tmp, m, y, n-x-1))
    
    free(tmp);

cdef inline signed char readField(signed char* f, int m, int x, int y):
    return f[y * m + x];

cdef inline void writeField(signed char* f, int m, int x, int y, signed char value):
    f[y * m + x] = value;

cdef signed char* initField(int m, int n, signed char v):
    cdef signed char* result = <signed char *> malloc(m * n * sizeof(signed char));
    
    for i in range(n * m):
        result[i] = v
    
    return result;

cdef int areFieldsEqual(int m, int n, signed char* fieldA, signed char* fieldB):
    cdef int eq = 1

    for i in range(n * m):
        if fieldA[i] != fieldB[i]:
            eq = 0
            break
    
    return eq

cdef void mirrorField(signed char* f, int m, int n):
    cdef int x, y;
    cdef signed char tmp;
    for y in range(n):
        for x in range(m / 2):
            tmp = readField(f, m, x, y)
            writeField(f, m, x, y, readField(f, m, m-1-x, y))
            writeField(f, m, m-1-x, y, tmp)

cdef void printField(int m, int n, signed char * field):
    s = ""
    
    for y in range(n):
        for x in range(m):
            s += "{0:.0f}".format(readField(field, m, x, y)) + " "
        s += "\n"
    
    logMsg(s)

cdef void rotateField(signed char* f, int m, int n):
    cdef signed char* tmp = initField(m, n, 0);
    memcpy(tmp, f, m * n * sizeof(signed char));
        
    cdef int x, y
    for y in range(n):
        for x in range(m):
            writeField(f, m, x, y, readField(tmp, m, y, n-1-x))
    
    free(tmp);
    
cdef void printFloatField(int m, int n, float* field):
    s = ""
    
    for y in range(n):
        for x in range(m):
            s += "{0:.4f}".format(readFloatField(field, m, x, y)) + " "
        s += "\n"
    
    logMsg(s)
