# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
# distutils: extra_link_args = -std=c++11

from impls.solved.pons.Solver cimport Solver

cdef class PascalPonsSolver:
    cdef Solver* c_solver

    def __cinit__(self):
        self.c_solver = new Solver()

    def __dealloc__(self):
        del self.c_solver

    def solve(self, moves):
        # do NOT call reset(). It limits speed on memsetting the transposition table to zero for no reason at all.
        result = self.c_solver.solve(moves.encode("UTF8"))
        if result == -4242:
            raise Exception("Invalid input: " + moves + "!")
        return result

    def loadBook(self, path):
        self.c_solver.loadBook(path.encode("UTF8"))    
