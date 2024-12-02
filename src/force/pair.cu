#include "pair.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>
#define BLOCK_SIZE_FORCE 64

PairPot::PairPot(FILE*, char*, int num_types, const int number_of_atoms, int basis_size, double min_val, double max_val, int n_species) : r_cut(max_val)

{
    p_Basis = new BasisChebyshev(basis_size, min_val, max_val, n_species);
}

static __global__ void calc_efs( 
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
  double* g_fz
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
            p_Basis->Calc(d12);
            for (int i = 0; i < p_Basis->size; ++i) {
                int pair_idx = i + type2 * p_Basis->size + type1 * p_Basis->size * p_Basis->n_species;
                en += 0.5 * rad_coeffs[pair_idx] * p_Basis->getVal(i) * pow((r_cut - d12), 2);
            }
            en_sum += en;
            double f12x = 0;
            double f12y = 0;
            double f12z = 0;
            p_Basis->CalcDers(x12);
            for (int i = 0; i < p_Basis->size; ++i) {
                int pair_idx = i + type2 * p_Basis->size + type1 * p_Basis->size * p_Basis->n_species;
                f12x += 0.5 * rad_coeffs[pair_idx] * (p_Basis->getDer(i)*pow((r_cut - x12), 2) + p_Basis->getVal(i) * (r_cut*r_cut - 
                2*r_cut - 2*x12))
            }
            p_Basis->CalcDers(y12);
            for (int i = 0; i < p_Basis->size; ++i) {
                int pair_idx = i + type2 * p_Basis->size + type1 * p_Basis->size * p_Basis->n_species;
                f12y += 0.5 * rad_coeffs[pair_idx] * (p_Basis->getDer(i)*pow((r_cut - y12), 2) + p_Basis->getVal(i) * (r_cut*r_cut - 
                2*r_cut - 2*y12))
            }
            p_Basis->CalcDers(z12);
            for (int i = 0; i < p_Basis->size; ++i) {
                int pair_idx = i + type2 * p_Basis->size + type1 * p_Basis->size * p_Basis->n_species;
                f12y += 0.5 * rad_coeffs[pair_idx] * (p_Basis->getDer(i)*pow((r_cut - z12), 2) + p_Basis->getVal(i) * (r_cut*r_cut - 
                2*r_cut - 2*z12))
            }
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
        eam_data.cell_count,
        eam_data.cell_count_sum,
        eam_data.cell_contents,
        eam_data.NN,
        eam_data.NL);
#ifdef USE_FIXED_NEIGHBOR
  }
#endif

    calc_efs<0><<<grid_size, BLOCK_SIZE_FORCE>>>(
      number_of_atoms,
      N1,
      N2,
      box,
      eam_data.NN.data(),
      eam_data.NL.data(),
      type.data(),
      eam_data.Fp.data(),
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

