#include <vector>
#include <string>

class AnyBasis {
    public:
        //std::vector<double> vals;
        //std::vector<double> ders;
        //double* vals;
        //double* ders;
        int size;
        int n_species;
        double min_val;
        double max_val;
        double scaling = 1.0;
        AnyBasis(int size_, double min_val_, double max_val_, int n_species_);
        AnyBasis(const std::string& filename);
        //virtual __device__ void Calc(double val) = 0;
        //virtual __device__ void CalcDers(double val_x) = 0;
        //inline __device__ double getVal(int i) {return vals[i];}
        //inline __device__ double getDer(int i) {return ders[i];}
        ~AnyBasis();
};


class BasisChebyshev : public AnyBasis {
    public:
        BasisChebyshev(int size_, double min_val_, double max_val_, int n_species_) : AnyBasis(size_, min_val_, max_val_, n_species_) {};
        BasisChebyshev(const std::string& filename) : AnyBasis(filename) {};
        //void __device__ Calc(double val) override;
        //void __device__ CalcDers(double val_x) override;
};
