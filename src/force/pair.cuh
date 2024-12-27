#pragma once
#include "potential.cuh"
#include "basis.cuh"
#include "utilities/gpu_vector.cuh"
#include <stdio.h>

struct PairPot_Data {
  GPU_Vector<int> NN, NL;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
};

class PairPot : public Potential {
public:
    using Potential::compute;
    AnyBasis* p_Basis = nullptr;
    std::vector<double> rad_coeffs;
    double rc = 0;
    PairPot(FILE*, char*, int num_types, const int number_of_atoms, int basis_size, double min_val, double max_val, int n_species);
    PairPot(const std::string& filename, const int number_of_atoms);
    void Load(const std::string& filename);
    virtual ~PairPot(void) {};
    virtual void compute(
        Box& box,
        const GPU_Vector<int>& type,
        const GPU_Vector<double>& position,
        GPU_Vector<double>& potential,
        GPU_Vector<double>& force,
        GPU_Vector<double>& virial);
  //void initialize_eam2004zhou(FILE*, int num_types);
  //void initialize_eam2006dai(FILE*); 
protected:
  int potential_model;
  PairPot_Data pp_data;
};