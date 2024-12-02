#include "basis.cuh"
#include <iostream>


AnyBasis::AnyBasis(int size_, double min_val_, double max_val_, int n_species_) : size(size_), min_val(min_val_), max_val(max_val_), 
    n_species(n_species_) {
        std::cout << "AnyBasis initialized" << std::endl;
        vals.resize(size);
        ders.resize(size);
    }

void BasisChebyshev::Calc(double val) {

    double ksi = (2 * val - (min_val + max_val)) / (max_val - min_val);

    vals[0] = scaling * 1;
    vals[1] = scaling * ksi;
    for (int i = 2; i < size; i++) {
        vals[i] = 2 * ksi * vals[i - 1] - vals[i - 2];
    }

}

void BasisChebyshev::CalcDers(double val)
{

    BasisChebyshev::Calc(val);

    double mult = 2.0 / (max_val - min_val);
    double ksi = (2 * val - (min_val + max_val)) / (max_val - min_val);

    ders[0] = scaling * 0;
    ders[1] = scaling * mult;
    for (int i = 2; i < size; i++) {
        ders[i] = 2 * (mult * vals[i - 1] + ksi * ders[i - 1]) - ders[i - 2];
    }
}