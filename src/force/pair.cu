#include "pair.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>
#define BLOCK_SIZE_FORCE 64

PairPot::PairPot(FILE*, char*, int num_types, const int number_of_atoms, int basis_size, double min_val, double max_val, int n_species)

{
    r_cut = max_val;
    p_Basis = new BasisChebyshev(basis_size, min_val, max_val, n_species);
    pp_data.NN.resize(number_of_atoms);
    pp_data.NL.resize(number_of_atoms * 400); // very safe for EAM
    pp_data.cell_count.resize(number_of_atoms);
    pp_data.cell_count_sum.resize(number_of_atoms);
    pp_data.cell_contents.resize(number_of_atoms);
}

// static __device__ void calc_energy ( 
//   double dist,
//   double min_val,
//   double max_val,
//   int scaling, 
//   int type1, 
//   int type2,
//   int basis_size,
//   int n_species,
//   double* rad_coeffs,
//   double &en)

// {
//     double* vals[];
//     double ksi = (2 * dist - (min_val + max_val)) / (max_val - min_val);
//     vals[0] = scaling * 1;
//     vals[1] = scaling * ksi;
//     for (int i = 2; i < basis_size; i++) {
//         vals[i] = 2 * ksi * vals[i - 1] - vals[i - 2];
//     }
//     for (int i = 0; i < basis_size; ++i) {
//         int pair_idx = i + type2 * basis_size + type1 * basis_size * n_species;
//         en += 0.5 * rad_coeffs[pair_idx] * vals[i] * pow((max_val - dist), 2);
//     }

// }

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

// static __device__ void calc_energy (
//   AnyBasis* p_Basis, 
//   double dist,
//   double r_cut, 
//   int type1, 
//   int type2,
//   std::vector<double> rad_coeffs,
//   double &en)

// {
//     p_Basis->Calc(dist);
//     for (int i = 0; i < p_Basis->size; ++i) {
//         int pair_idx = i + type2 * p_Basis->size + type1 * p_Basis->size * p_Basis->n_species;
//         en += 0.5 * rad_coeffs[pair_idx] * p_Basis->getVal(i) * pow((r_cut - dist), 2);
//     }

// }

// static __device__ void calc_force ( 
//   double dist,
//   double r_cut, 
//   int type1, 
//   int type2,
//   std::vector<double> rad_coeffs,
//   double &f)

// {
//     p_Basis->CalcDers(dist);
//     for (int i = 0; i < p_Basis->size; ++i) {
//         int pair_idx = i + type2 * p_Basis->size + type1 * p_Basis->size * p_Basis->n_species;
//         f += 0.5 * rad_coeffs[pair_idx] * (p_Basis->getDer(i)*pow((r_cut - dist), 2) + p_Basis->getVal(i) * (r_cut*r_cut - 
//         2*r_cut - 2*dist));
//     }

// }

static __global__ void calc_efs(
  AnyBasis* p_Basis,
  double* rad_coeffs,
  double r_cut, 
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
        for (int i1 = 0; i1 < NN; ++i1) {
            int n2 = g_NL[n1 + N * i1];
            int type2 = g_type[n2];
            double x12 = g_x[n2] - x1;
            double y12 = g_y[n2] - y1;
            double z12 = g_z[n2] - z1;
            apply_mic(box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double en = 0.0;
            double f12x = 0;
            double f12y = 0;
            double f12z = 0;
            // calc_energy(p_Basis, d12, r_cut, type1, type2, rad_coeffs, en);
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

    const int number_of_atoms = type.size();
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

#ifdef USE_FIXED_NEIGHBOR
  static int num_calls = 0;
#endif
#ifdef USE_FIXED_NEIGHBOR
  if (num_calls++ == 0) {
#endif
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
    
    AnyBasis* cp_Basis;
    cudaMalloc((void **)&cp_Basis, sizeof(AnyBasis));
    cudaMemcpy(cp_Basis, p_Basis, sizeof(AnyBasis), cudaMemcpyHostToDevice);
    double* host_vals = new double[p_Basis->size];
    double* host_ders = new double[p_Basis->size];
    cudaMalloc((void **)&host_vals, sizeof(double)*p_Basis->size);
    cudaMalloc((void **)&host_ders, sizeof(double)*p_Basis->size);
    cudaMemcpy(host_vals, p_Basis->vals, sizeof(double)*p_Basis->size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(cp_Basis->vals), &host_vals, sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(host_ders, p_Basis->ders, sizeof(double)*p_Basis->size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(cp_Basis->ders), &host_ders, sizeof(double *), cudaMemcpyHostToDevice);
    double* p_rad_coeffs = new double[rad_coeffs.size()];
    cudaMalloc((void **)&p_rad_coeffs, sizeof(double)*rad_coeffs.size());
    cudaMemcpy(p_rad_coeffs, rad_coeffs.data(), sizeof(double)*rad_coeffs.size(), cudaMemcpyHostToDevice);
    calc_efs<<<grid_size, BLOCK_SIZE_FORCE>>>(
      cp_Basis,
      p_rad_coeffs,
      r_cut,
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

