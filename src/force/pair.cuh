#pragma once
#include "potential.cuh"
#include "basis.cuh"
#include "utilities/gpu_vector.cuh"
#include <stdio.h>

class PairPot : public Potential {
public:
    using Potential::compute;
    AnyBasis* p_Basis = nullptr;
    vector<double> rad_coeffs;
    double r_cut = 0;
    PairPot(FILE*, char*, int num_types, const int number_of_atoms, int basis_size, double min_val, double max_val);
    virtual ~Pair(void);
    virtual void compute(
        Box& box,
        const GPU_Vector<int>& type,
        const GPU_Vector<double>& position,
        GPU_Vector<double>& potential,
        GPU_Vector<double>& force,
        GPU_Vector<double>& virial);
  //void initialize_eam2004zhou(FILE*, int num_types);
  //void initialize_eam2006dai(FILE*); 
}