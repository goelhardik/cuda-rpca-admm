#ifndef _rpca_admm_kernel_h
#define _rpca_admm_kernel_h

#include <cuda.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>

#define PRECESION   1.0e-10

template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const { 
            return x * x;
        }
};

template<typename T>
struct absolute_value
{
  __host__ __device__ T operator()(const T &x) const {
        return x < T(0) ? -x : x;
    }
};

struct diffsq{
  __host__ __device__ float operator()(float a, float b)
  {
    return (b-a)*(b-a);
  }
};

__global__ void nodiag_normalize(float *A, float *I, int n, int i);
__global__ void diag_normalize(float *A, float *I, int n, int i);
__global__ void gaussjordan(float *A, float *I, int n, int i);
__global__ void set_zero(float *A, float *I, int n, int i);
#endif
