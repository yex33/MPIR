# Proposed Report Outline

## Front Matter

* Title Page (from template)
* Abstract
* Acknowledgements
* Table of Contents
* List of Figures / Tables

## Chapter 1: Introduction

* Background on large-scale linear systems in scientific computing
* Motivation for using iterative solvers (GMRES, IR) for sparse matrices
* Problem statement and objectives
    * Efficient solution of large sparse symmetric systems
    * Exploiting mixed precision for performance
    * Fine-grained parallel ILU preconditioning
    * C++ template and concept-based solver design
* Contributions of this work
* Report organization

## Chapter 2: Literature Review

* Iterative Solvers
    * Overview of Krylov subspace methods
    * GMRES and restarted GMRES
    * Iterative refinement and history of mixed precision methods
* Preconditioning
    * ILU factorization, IC factorization, fill-in levels
    * Parallel ILU algorithms (fine-grained vs level-scheduling)
* Mixed Precision Computing
    * Role in modern HPC (e.g., GPUs, CPUs with wide SIMD)
    * Previous work on iterative refinement with low precision factorizations

## Chapter 3: System Requirements and Design (from SRS)

* Problem domain recap (aligned with SRS)
* Functional requirements: solver features, interfaces, inputs/outputs
* Non-functional requirements: performance, scalability, precision handling, numerical stability
* High-level system architecture: solver modules (GMRES, IR, ILU, utilities)
* Design goals from SRS mapped to implementation

## Chapter 4: Mathematical Formulation

* GMRES algorithm derivation
    * Krylov subspace basis construction
    * Arnoldi process
    * Givens rotations for Hessenberg system
* Iterative Refinement with mixed precision
    * Error correction model
    * Role of factorization precision (UF), working precision (UW), residual precision (UR)
* Incomplete LU (ILU(k)) and IC factorization
    * Fill-in rules, stability considerations
    * Nonlinear equations for fine-grained parallel ILU sweeps
* Sparse matrix storage formats
    * CSR, CSC, symmetry exploitation
    * Conversion between formats

## Chapter 5: Implementation

* C++ Template and Concepts Design
    * Refinable concept (UF, UW, UR precisions)
    * Generic utilities (VectorMultiply, MatrixMultiply, etc.)
* GMRES Implementation
    * Preconditioned GMRES
    * Restart strategies, residual monitoring
* Iterative Refinement Driver
    * Mixed precision workflow
    * Solver pipeline: residual compute → GMRES solve → correction step
* ILU Preconditioner
    * Fine-grained parallelization approach
    * SparseLDotU kernel, nonlinear sweeps, fill-in management
* Triangular Solves
    * Forward/backward substitution for LU in CSC/CSR hybrid
* Parallelization
    * OpenMP scheduling strategies (dynamic, fine-grained loops)
    * Reduction strategies for residual norms

## Chapter 6: Experimental Evaluation

* Experimental setup
    * Hardware (CPU/GPU details, core counts, SIMD width)
    * Test matrices (symmetric sparse benchmarks, synthetic vs real)
* Experiments
    * Accuracy: residual reduction under different precisions
    * Performance: runtime scaling with matrix size
    * GMRES restart interval impact (20 vs 1000 Krylov size)
    * ILU fill-in level (k) effect on convergence and runtime
    * Strong/weak scaling with OpenMP threads
* Results & Discussion
    * Trade-offs between accuracy and speed
    * Conditions where residual stagnates or fluctuates
    * Impact of mixed precision vs uniform double precision

## Chapter 7: Discussion and Lessons Learned

* Effectiveness of mixed precision iterative refinement
* Stability issues and remedies (reorthogonalization, scaling)
* Parallel ILU observations (fill-in behavior, convergence to ILU(0))
* Code maintainability with C++ templates and concepts
* Limitations of current implementation

## Chapter 8: Conclusion and Future Work

* Summary of contributions
* Practical implications for large-scale HPC solvers
* Future directions:
    * GPU acceleration with CUDA/HIP/SYCL
    * Communication-avoiding GMRES
    * Adaptive precision refinement
    * Integration into existing scientific computing frameworks (PETSc, Trilinos)

## Appendices

* Code snippets (key kernels: TriangularSolve, SparseLDotU, PrecondGmres)
* Detailed algorithm pseudocode for ILU(k)
* Additional experimental plots
* SRS document copy/reference


- Assumed input matrices are SPD for simplicity and ease of implementation
- mimic interfaces of sparse solvers from Eigen. This includes:
    - A solver class where the template type parameters UF, UW, UR are specified, corresponding 
    to the factorization, working, and residual precisions given in the MP iterative 
    refinement algorithm
    - A compute method which accepts an input matrix and performs the ILU matrix
    factorization. Note that the class instance owns a copy of the input matrix. 
    Caller can move the matrix into the class instance.
    - A solve method which accepts an input vector and returns the numeric solution.
- make extensive use of C++ templates for mixed precision support, combined with variadic concepts to restrict
valid floating-point types and enforce non-decreasing machine epsilon for a series of types.
- extended the QD package (which provides quad and octuple precision floating point types):
complete the type info by adding proper constexpr std::numeric_limit, implement formatting interface to support std::print
- modular design by grouping matrix operations, such as sparse matrix-vector 
multiplication, dense vector multiplication, etc. in a separate module. These BLAS-like functions 
are full templated to support mixed precision arithmetic.
