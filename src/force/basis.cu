#include "basis.cuh"
#include <iostream>
#include <fstream>
#include <thrust/device_vector.h>


AnyBasis::AnyBasis(int size_, double min_val_, double max_val_, int n_species_) : size(size_), min_val(min_val_), max_val(max_val_), 
    n_species(n_species_) {
        //vals = new double[size];
        //ders = new double[size];
        std::cout << "AnyBasis initialized" << std::endl;
        //vals.resize(size);
        //ders.resize(size);
        
}
/*AnyBasis::~AnyBasis() {
  delete[] vals;
  delete[] ders;
}*/
AnyBasis::AnyBasis(const std::string& filename) 

{
  std::cout<<"initializing anybasis"<<std::endl;
  std::ifstream ifs(filename);
  std::string tmpstr;
  std::getline(ifs, tmpstr);
  std::getline(ifs, tmpstr);
  ifs >> tmpstr;
  std::cout<<tmpstr<<std::endl;
  if (tmpstr == "min_dist") {
    ifs.ignore(3);
    ifs >> min_val;
    ifs >> tmpstr;
  }
  if (tmpstr == "max_dist") {
    ifs.ignore(3);
    ifs >> max_val;
    ifs >> tmpstr;
  }
  if (tmpstr == "basis_size") {
    ifs.ignore(3);
    ifs >> size;
    ifs >> tmpstr;
  }
  if (tmpstr == "n_species") {
    ifs.ignore(3);
    ifs >> n_species;
  }
}

/*__device__ void Calc(thrust::device_vector<double> vals, double val, double min_val, double max_val, double scaling, int size) 
{

    //thrust::device_vector<double> vals;
    double ksi = (2 * val - (min_val + max_val)) / (max_val - min_val);

    vals[0] = scaling * 1;
    vals[1] = scaling * ksi;
    for (int i = 2; i < size; i++) {
        vals[i] = 2 * ksi * vals[i - 1] - vals[i - 2];
    }

}

__device__ void CalcDers(thrust::device_vector<double> vals, thrust::device_vector<double> ders, double val, double min_val, double max_val, double scaling, int size)
{

    //BasisChebyshev::Calc(val);

    double mult = 2.0 / (max_val - min_val);
    double ksi = (2 * val - (min_val + max_val)) / (max_val - min_val);

    ders[0] = scaling * 0;
    ders[1] = scaling * mult;
    for (int i = 2; i < size; i++) {
        ders[i] = 2 * (mult * vals[i - 1] + ksi * ders[i - 1]) - ders[i - 2];
    } 
}*/
