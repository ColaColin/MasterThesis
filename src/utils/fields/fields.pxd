cdef float readFloatField(float* f, int m, int x, int y)

cdef void writeFloatField(float* f, int m, int x, int y, float value)

cdef float* initFloatField(int m, int n, float v)

cdef void mirrorFloatField(float* f, int m, int n)

cdef void rotateFloatField(float* f, int m, int n)

cdef signed char readField(signed char* f, int m, int x, int y)

cdef void writeField(signed char* f, int m, int x, int y, signed char value)

cdef signed char* initField(int m, int n, signed char v)

cdef int areFieldsEqual(int m, int n, signed char* fieldA, signed char* fieldB)

cdef void mirrorField(signed char* f, int m, int n)

cdef void printField(int m, int n, signed char * field)

cdef void rotateField(signed char* f, int m, int n)

cdef void printFloatField(int m, int n, float* field)

