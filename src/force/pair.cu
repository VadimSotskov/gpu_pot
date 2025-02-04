#include "pair.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>
#include <iostream>
#include <thrust/device_vector.h>
#define BLOCK_SIZE_FORCE 64

PairPot::PairPot(FILE*, char*, int num_types, const int number_of_atoms, int basis_size, double min_val, double max_val, int n_species)

{
    pp_data.NN.resize(number_of_atoms);
    pp_data.NL.resize(number_of_atoms * 400); // very safe for EAM
    pp_data.cell_count.resize(number_of_atoms);
    pp_data.cell_count_sum.resize(number_of_atoms);
    pp_data.cell_contents.resize(number_of_atoms);
}

PairPot::PairPot(const std::string& filename, const int number_of_atoms) 
{
  Load(filename);
  pp_data.NN.resize(number_of_atoms);
  pp_data.NL.resize(number_of_atoms * 400); // very safe for EAM
  pp_data.cell_count.resize(number_of_atoms);
  pp_data.cell_count_sum.resize(number_of_atoms);
  pp_data.cell_contents.resize(number_of_atoms);
}

void PairPot::Load(const std::string& filename) 

{
  std::cout<<"LOADING POT"<<std::endl;
  std::ifstream ifs(filename);
  std::string tmpstring;
  ifs >> tmpstring;
  ifs.ignore (6);
  ifs >> tmpstring;
  std::cout<<"now string is"<<std::endl;
  std::cout<<tmpstring<<std::endl;
  if (tmpstring == "basis_type") {
    ifs.ignore(3);
    ifs >> tmpstring;
    std::cout<<tmpstring<<std::endl;
    if (tmpstring == "Chebyshev") {
      std::cout<<"Initializing basis cheb"<<std::endl;
      p_Basis = new BasisChebyshev(filename);
      ifs >> tmpstring;
    }
  }
  std::cout<<"Inited basis"<<std::endl;
  for (int i = 0; i < 5; ++i) std::getline(ifs, tmpstring);
  std::cout<<tmpstring<<std::endl;
  std::cout<<"Basis size is: "<<p_Basis->size<<std::endl;
  rad_coeffs.resize(p_Basis->size * pow(p_Basis->n_species, 2));
  for (int i = 0; i < rad_coeffs.size(); ++i) {
      ifs >> rad_coeffs[i];
  }
  std::cout<<"COEFFS FILLED"<<std::endl;
  rc = p_Basis->max_val;
    
    
}

static __device__ void calc_energy ( 
    double dist,
    double min_val,
    double max_val,
    int scaling, 
    int type1, 
    int type2,
    int basis_size,
    int n_species,
    double* rad_coeffs,
    double* vals,
    double &en)

{
    //double vals = new double[basis_size];
    double mult = 2.0 / (max_val - min_val);
    double ksi = (2 * dist - (min_val + max_val)) / (max_val - min_val);
    vals[0] = scaling * (1 * (dist - max_val) * (dist - max_val));
    vals[1] = scaling * (ksi * (dist - max_val) * (dist - max_val));
    for (int i = 2; i < basis_size; i++) {
        vals[i] = 2 * ksi * vals[i - 1] - vals[i - 2];
    }
    //printf("DISTANCE: %f\n", ksi);
    //for (int i = 0; i < basis_size; ++i) printf("%f\n", vals[i]);
    for (int i = 0; i < basis_size; ++i) {
        int pair_idx = i + type2 * basis_size + type1 * basis_size * n_species;
        en += rad_coeffs[pair_idx] * vals[i];
        //en += 0.5 * rad_coeffs[pair_idx] * vals[i] * pow((max_val - dist), 2);
    }
    //delete[] vals;

}

// static __device__ void calc_force ( 
//   double dist,
//   double min_val,
//   double max_val,
//   int scaling, 
//   int type1, 
//   int type2,
//   int basis_size,
//   int n_species,
//   double* rad_coeffs,
//   double &f)

// {
//     p_Basis->CalcDers(dist);
//     for (int i = 0; i < p_Basis->size; ++i) {
//         int pair_idx = i + type2 * p_Basis->size + type1 * p_Basis->size * p_Basis->n_species;
//         f += 0.5 * rad_coeffs[pair_idx] * (p_Basis->getDer(i)*pow((r_cut - dist), 2) + p_Basis->getVal(i) * (r_cut*r_cut - 
//         2*r_cut - 2*dist));
//     }

// }

/*static __device__ void Calc(thrust::device_vector<double> vals, double val, double min_val, double max_val, double scaling, int size) 
{

    //thrust::device_vector<double> vals;
    double ksi = (2 * val - (min_val + max_val)) / (max_val - min_val);

    vals[0] = scaling * 1;
    vals[1] = scaling * ksi;
    for (int i = 2; i < size; i++) {
        vals[i] = 2 * ksi * vals[i - 1] - vals[i - 2];
    }

}*/

/*static __device__ void CalcDers(thrust::device_vector<double> vals, thrust::device_vector<double> ders, double val, double min_val, double max_val, double scaling, int size)
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

/*static __device__ void calc_energy (
  double min_val,
  double max_val,
  double scaling,
  int size,
  int n_species, 
  double dist,
  double rс, 
  int type1, 
  int type2,
  double* rad_coeffs,
  double &en)

{
    //thrust::device_vector<double> vals;
    //thrust::device_vector<double> ders;
    //Calc(vals, dist, min_val, max_val, scaling, size);
    //CalcDers(vals, ders, dist, min_val, max_val, scaling, size);    
    //p_Basis->Calc(dist);
    for (int i = 0; i < size; ++i) {
        int pair_idx = i + type2 * size + type1 * size * n_species;
        //en += 0.5 * rad_coeffs[pair_idx] * vals[i] * pow((rс - dist), 2);
    }

}*/

static __device__ void calc_force ( 
    double dist,
    double min_val,
    double max_val,
    int scaling, 
    int type1, 
    int type2,
    int basis_size,
    int n_species,
    double* rad_coeffs,
    double* vals,
    double* ders,
    double &f)


{
    double mult = 2.0 / (max_val - min_val);
    double ksi = (2 * dist - (min_val + max_val)) / (max_val - min_val);

    ders[0] = scaling * 0;
    ders[1] = scaling * mult;
    for (int i = 2; i < basis_size; i++) {
        ders[i] = 2 * (mult * vals[i - 1] + ksi * ders[i - 1]) - ders[i - 2];
    } 

    for (int i = 0; i < basis_size; ++i) {
        int pair_idx = i + type2 * basis_size + type1 * basis_size * n_species;
        f += 0.5 * rad_coeffs[pair_idx] * (ders[i]*pow((max_val - dist), 2) + vals[i] * (max_val*max_val -  2*max_val - 2*dist));
    }

}

static __global__ void calc_efs(
  double min_val,
  double max_val,
  double scaling,
  int size,
  int n_species,  
  double* rad_coeffs,
  double* vals,
  double* ders,
  double rс, 
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)

{
    //for (int i = 0; i < 24; ++i) printf("%f\n", rad_coeffs[i]);
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; //particle index
    int type1 = g_type[n1];

    if (n1 < N2) {
        int NN = g_NN[n1];

        double x1 = g_x[n1];
        double y1 = g_y[n1];
        double z1 = g_z[n1];

        // Calculate the expansion for neighborhood
        double en_sum = 0;
        double s_fx = 0;
        double s_fy = 0;
        double s_fz = 0;
        printf("NN: %d\n", NN);
        for (int i1 = 0; i1 < NN; ++i1) {
            int n2 = g_NL[n1 + N * i1];
            int type2 = g_type[n2];
            double x12 = g_x[n2] - x1;
            double y12 = g_y[n2] - y1;
            double z12 = g_z[n2] - z1;
            apply_mic(box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            //printf("DISTANCE: %f\n", d12);
            double en = 0.0;
            double f12x = 0;
            double f12y = 0;
            double f12z = 0;
            //printf("MIN VAL: %f\n", min_val);
            //printf("MAX VAL: %f\n", max_val);
            //printf("SCALING: %f\n", scaling);
            calc_energy(d12, min_val, max_val, scaling, type1, type2, size, n_species, rad_coeffs, vals, en);
            calc_force(d12, min_val, max_val, scaling, type1, type2, size, n_species, rad_coeffs, vals, ders, f12x);
            calc_force(d12, min_val, max_val, scaling, type1, type2, size, n_species, rad_coeffs, vals, ders, f12y);
            calc_force(d12, min_val, max_val, scaling, type1, type2, size, n_species, rad_coeffs, vals, ders, f12z);
            // calc_force(p_Basis, x12, r_cut, type1, type2, rad_coeffs, f12x);
            // calc_force(p_Basis, x12, r_cut, type1, type2, rad_coeffs, f12y);
            // calc_force(p_Basis, x12, r_cut, type1, type2, rad_coeffs, f12z);
            en_sum += en;
            s_fx += f12x;
            s_fy += f12y;
            s_fz += f12z;

        }


        g_pe[n1] += en_sum; // many-body potential energy
        g_fx[n1] = s_fx;
        g_fy[n1] = s_fy;
        g_fz[n1] = s_fz;
        //printf("ENERGY: %f\n", en_sum);
        //printf("FX: %f\n", s_fx);
        //printf("FY: %f\n", s_fy);
        //printf("FZ: %f\n", s_fz);


  }
}

void PairPot::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
    //std::cout<<"in compute"<<std::endl;
    const int number_of_atoms = type.size();
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

#ifdef USE_FIXED_NEIGHBOR
  static int num_calls = 0;
#endif
#ifdef USE_FIXED_NEIGHBOR
  if (num_calls++ == 0) {
#endif
    //std::cout<<"finding neighbor"<<std::endl;
    //std::cout<<N1<<" "<<N2<<std::endl;
    //std::cout<<rc<<std::endl;
    //std::cout<<number_of_atoms<<std::endl;
    find_neighbor(
        N1,
        N2,
        rc,
        box,
        type,
        position_per_atom,
        pp_data.cell_count,
        pp_data.cell_count_sum,
        pp_data.cell_contents,
        pp_data.NN,
        pp_data.NL);
#ifdef USE_FIXED_NEIGHBOR
  }
#endif
    //std::cout<<"neighbor found"<<std::endl;
    //AnyBasis* cp_Basis;
    //printf("%f\n", cp_Basis->max_val);
    //cudaMalloc((void **)&cp_Basis, sizeof(AnyBasis));
    //cudaMemcpy(cp_Basis, p_Basis, sizeof(AnyBasis), cudaMemcpyHostToDevice);
    //printf("%f\n", cp_Basis->max_val);
    //double* host_vals = new double[p_Basis->size];
    //double* host_ders = new double[p_Basis->size];
    //cudaMalloc((void **)&host_vals, sizeof(double)*p_Basis->size);
    //cudaMalloc((void **)&host_ders, sizeof(double)*p_Basis->size);
    //cudaMemcpy(host_vals, p_Basis->vals, sizeof(double)*p_Basis->size, cudaMemcpyHostToDevice);
    //for(int i = 0; i < p_Basis->size; ++i) {
    //    printf("%f\n",host_vals[i]);
    //}
    //std::cout<<"copied!"<<std::endl;
    //cudaMemcpy(&(cp_Basis->vals), &host_vals, sizeof(double *), cudaMemcpyHostToDevice);
    //std::cout<<"copied1!"<<std::endl;
    //cudaMemcpy(host_ders, p_Basis->ders, sizeof(double)*p_Basis->size, cudaMemcpyHostToDevice);
    //std::cout<<"copied basis1"<<std::endl;
    //cudaMemcpy(&(cp_Basis->ders), &host_ders, sizeof(double *), cudaMemcpyHostToDevice);
    //double* p_rad_coeffs = new double[rad_coeffs.size()];
    //std::cout<<"RAD COEFFS SIZE: "<<rad_coeffs.size()<<std::endl;
    //for(int i = 0; i < rad_coeffs.size(); ++i) std::cout<<rad_coeffs[i]<<std::endl;
    //cudaMalloc((void **)&p_rad_coeffs, sizeof(double)*rad_coeffs.size());
    //cudaMemcpy(p_rad_coeffs, rad_coeffs.data(), sizeof(double)*rad_coeffs.size(), cudaMemcpyHostToDevice);
    thrust::device_vector<double> c_coeffs(rad_coeffs.size());
    thrust::device_vector<double> c_basis_vals(p_Basis->size);
    thrust::device_vector<double> c_basis_ders(p_Basis->size);
    for (int i = 0; i < rad_coeffs.size(); ++i)
        c_coeffs[i] = rad_coeffs[i];
    double* p_c_coeffs = thrust::raw_pointer_cast(c_coeffs.data());
    double* p_c_basis_vals = thrust::raw_pointer_cast(c_basis_vals.data());
    double* p_c_basis_ders = thrust::raw_pointer_cast(c_basis_ders.data());
    //std::cout<<"copied to gpu"<<std::endl;
    calc_efs<<<grid_size, BLOCK_SIZE_FORCE>>>(
      p_Basis->min_val,
      p_Basis->max_val,
      p_Basis->scaling,
      p_Basis->size,
      p_Basis->n_species,
      p_c_coeffs,
      p_c_basis_vals,
      p_c_basis_ders,
      rc,
      number_of_atoms,
      N1,
      N2,
      box,
      pp_data.NN.data(),
      pp_data.NL.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + number_of_atoms,
      position_per_atom.data() + number_of_atoms * 2,
      force_per_atom.data(),
      force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
}

