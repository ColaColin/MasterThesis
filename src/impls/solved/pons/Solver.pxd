from libcpp.string cimport string

cdef extern from "Solver.cpp":
    pass

cdef extern from "Solver.hpp" namespace "GameSolver::Connect4":
    cdef cppclass Solver:
        Solver() except +
        int solve(string)
        unsigned long long getNodeCount()
        void reset()
        void loadBook(string)