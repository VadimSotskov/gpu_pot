#include <vector>

class AnyBasis {
    public:
        std::vector<double> vals;
        std::vector<double> ders;
        int size;
        int n_species;
        double min_val;
        double max_val;
        double scaling = 1.0;
        AnyBasis(int size_, double min_val_, double max_val_, int n_species_);
        virtual void Calc(double val);
        virtual void CalcDers(double val_x);
        inline double getVal(int i) {return vals[i];}
        inline double getDer(int i) {return ders[i];}
};


class BasisChebyshev : public AnyBasis {
    public:
        BasisChebyshev(int size_, double min_val_, double max_val_, int n_species_) : AnyBasis(size_, min_val_, max_val_, n_species_) {};
        void Calc(double val) override;
        void CalcDers(double val_x) override;
};