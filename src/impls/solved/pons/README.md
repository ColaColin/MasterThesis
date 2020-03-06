# Connect 4 Game Solver

This C++ source code is published under AGPL v3 license.

Read the associated [step by step tutorial to build a perfect Connect 4 AI](http://blog.gamesolver.org) for explanations.

The code was modified in a minimal way to simplify the Solver interface so that writing pons.pyx and Solver.pxd was easier.
This allows to call the solver directly from python with no subprocess-line-buffered overhead.