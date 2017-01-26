#include <stdio.h>
#include "rpca_admm_kernel.cuh"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "cublas.h"
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define BLOCK_SIZE 8 

typedef struct GPUInfo
{
    unsigned int MAX_GRID_SIZE;
    unsigned int MAX_BLOCK_SIZE;
}GPUInfo;

typedef struct ADMM_para
{
    float* iter_time;
    unsigned int MAX_ITER;          // MAX_ITER
    float abstol;
    float reltol;
    float lambda;
} ADMM_para;

typedef struct RPCA_ADMM 
{
    int m;
    int n;
    int N;

    // we'll call our matrix A
    float* A;                       // row major order
    float *X1, *X2, *X3;

    float g2;
    float g3;
    bool SAVEFILE;
} RPCA_ADMM;

/*
 * Thrust update functions for tranformations
 */
struct B_update {
    const float N_loc;

    B_update(float _N_loc) : N_loc(_N_loc) {}

    __host__ __device__
    float operator()(thrust::tuple<float,float,float,float,float> t) {
        float x1, x2, x3, a, u;
        thrust::tie(x1, x2, x3, a, u) = t;
        return (((x1 + x2 + x3) / 3.0f) - (a / N_loc) + u);
    }
};

struct avg_cal {
    const float N_loc;

    avg_cal(float _N_loc) : N_loc(_N_loc) {}

    __host__ __device__
    float operator()(thrust::tuple<float,float,float,float> t) {
        float x1, x2, x3, a;
        thrust::tie(x1, x2, x3, a) = t;
        return (((x1 + x2 + x3) / 3.0f) - (a / N_loc));
    }
};

struct X1_update {
    const float lamb;

    X1_update(float _lamb) : lamb(_lamb) {}

    __host__ __device__
    float operator()(const float &x1, const float &u) {
        return ((1.0f / (1.0f + lamb)) * (x1 - u));
    }
};


struct X2_update {
    const float temp;

    X2_update(float _temp) : temp(_temp) {}

    __host__ __device__
    float operator()(const float &x2, const float &u) {
        float v = x2 - u;
        float ans = (v - temp > 0.0f? v - temp : 0.0f) - (-v - temp > 0.0f? -v - temp : 0.0f);
        return ans;
    }
};

struct prox_l1 {
    const float temp;

    prox_l1(float _temp) : temp(_temp) {}

    __host__ __device__
    float operator()(const float &s) {
        float v = s;
        float ans = (v - temp > 0.0f? v - temp : 0.0f) - (-v - temp > 0.0f? -v - temp : 0.0f);
        return ans;
    }
};


struct diag_index : public thrust::unary_function<int,int>
{
    const int rows;
    
    diag_index(int rows) : rows(rows){}

    __host__ __device__
    int operator()(const int index) const {
        return (index*rows + (index%rows));
    }

};

template<typename V>
void print_matrix(const V& mat, int rows, int cols) {
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            std::cout << mat[i*cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    // Destroy the handle
    cublasDestroy(handle);
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void
fast_svd_routine(float *d_A, unsigned int m, unsigned int n, float *d_svd_U, float *d_svd_S, float *d_svd_VH, signed char jobu, signed char jobv)
{
        float *U = new float[m*n];
        // svd - economy version code coming in
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        const float alph= 1;
        const float bet = 0;
        const float* alpha = &alph;
        const float* beta = &bet;

        int *d_pivot;  
        float *h_X3chol;
        float *d_X3chol, *d_X3_InvR, *d_X3_Q, *d_X3_U;
        float *work;   
        int *devInfo;
        int work_size = 0;

        h_X3chol = (float *) malloc(n*n*sizeof(float));
        cudaMalloc(&d_X3chol, n*n*sizeof(float));
        cudaMalloc(&d_X3_InvR, n*n*sizeof(float));
        cudaMalloc(&d_X3_Q, m*n*sizeof(float));
        cudaMalloc(&d_X3_U, m*n*sizeof(float));
        cudaMalloc(&d_pivot, n*sizeof(int));

        // multiply A.T * A which serves as in input to Cholesky
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, alpha, d_A, m, d_A, m, beta, d_X3chol, n);
        cublasDestroy(handle);
    
        // Cholesky Code coming in  
        gpuErrchk(cudaMalloc(&devInfo, sizeof(int)));

        cusolverDnHandle_t solver_handle;
        cusolverDnCreate(&solver_handle);
        
        cusolverDnSpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, n, d_X3chol, n, &work_size);
        gpuErrchk(cudaMalloc(&work, work_size * sizeof(float)));

        cusolverDnSpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, n, d_X3chol, n, work, work_size, devInfo);
        cudaDeviceSynchronize();
        cusolverDnDestroy(solver_handle);
        cudaMemcpy(h_X3chol, d_X3chol, n*n*sizeof(float), cudaMemcpyDeviceToHost);
        // cublas does not zero out the lower triangle
        // zero it in CPU
        for (int i = 0; i < n; i++) {
            for (int j = i+1; j < n; j++) {
                h_X3chol[i*n + j] = 0;
            }
        }
        cudaMemcpy(d_X3chol, h_X3chol, n*n*sizeof(float), cudaMemcpyHostToDevice);

        
        float *h_I;
        h_I = (float *) malloc(n*n*sizeof(float));
        for (int i = 0; i<n; i++) {
            for (int j = 0; j<n; j++) {
                if (i == j) 
                    h_I[i*n + i] = 1.0;
                else 
                    h_I[i*n + j] = 0.0;
            }
        }

        unsigned int blocksize = BLOCK_SIZE;
        dim3 threadsPerBlock(blocksize, blocksize);
        dim3 numBlocks((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
         
        cudaStream_t s1;
        cudaStreamCreate(&s1);

        float *d_X3chol_copy;
        cudaMalloc(&d_X3chol_copy, n*n*sizeof(float));
        cudaMemcpy(d_X3_InvR, h_I, n*n*sizeof(float), cudaMemcpyHostToDevice);
        
        cudaMemcpyAsync(d_X3chol_copy, d_X3chol, n*n*sizeof(float), cudaMemcpyDeviceToDevice, s1);
        
        for (int i = 0; i < n; i++)
		{
			nodiag_normalize << <numBlocks, threadsPerBlock >> >(d_X3chol, d_X3_InvR, n, i);
			diag_normalize << <numBlocks, threadsPerBlock >> >(d_X3chol, d_X3_InvR, n, i);
			gaussjordan << <numBlocks, threadsPerBlock >> >(d_X3chol, d_X3_InvR, n, i);
			set_zero << <numBlocks, threadsPerBlock >> >(d_X3chol, d_X3_InvR, n, i);
		}
        

        cublasHandle_t handle_Q;
        cublasCreate(&handle_Q);
        // Q = A * inv(R)
        cublasSgemm(handle_Q, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, alpha, d_A, m, d_X3_InvR, n, beta, d_X3_Q, m);
        cublasDestroy(handle_Q);
        
        // svd(R) coming in
        cusolverDnHandle_t solver_handle_svd;
        cusolverDnCreate(&solver_handle_svd);
        cusolverDnSetStream(solver_handle_svd, s1);
        work_size = 0;
        cusolverDnSgesvd_bufferSize(solver_handle_svd, n, n, &work_size);
        
        float *work_svd;
        gpuErrchk(cudaMalloc(&work_svd, work_size * sizeof(float)));
        // --- CUDA SVD execution
        cusolverDnSgesvd(solver_handle_svd, jobu, jobv, n, n, d_X3chol_copy, n, d_svd_S, d_svd_U, n, d_svd_VH, n, work_svd, work_size, NULL, devInfo);
        cudaDeviceSynchronize();

        cusolverDnDestroy(solver_handle_svd);
        cudaStreamDestroy(s1);
        cublasHandle_t handle_U;
        cublasCreate(&handle_U);
        // U = Q * UR
        cublasSgemm(handle_U, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, alpha, d_X3_Q, m, d_svd_U, n, beta, d_X3_U, m);
        cublasDestroy(handle_U);
        cudaMemcpy(d_svd_U, d_X3_U, sizeof(float)*m*n, cudaMemcpyDeviceToDevice);


        cudaFree(d_X3chol);
        cudaFree(d_X3_InvR);
        cudaFree(d_X3_Q);
        cudaFree(d_X3_U);
        cudaFree(d_pivot);
        cudaFree(devInfo);
        cudaFree(work);
        cudaFree(work_svd);
}



/*
 * ADMM for RPCA. 
 * All matrices are in column major order
 */
void cuda_rpca_admm(RPCA_ADMM * &rpca, ADMM_para* &admm_param, GPUInfo* gpu_info)
{
    float *U;     // host (boyd code)

    double s_norm, r_norm;

    // device
    float *d_A, *d_X_1, *d_X_2, *d_X_3, *d_X_3_tmp, *d_U;
    float *d_z, *d_z_old;
    float *d_svd_U, *d_svd_S, *d_svd_VH, *d_svd_S_diag, *d_svd_temp;

    unsigned int m,n,N;
    m = rpca->m;
    n = rpca->n;
    N = rpca->N;

    unsigned long int size = m*n;
    U = new float[size];
    float fill_value = 0.0f;

    // allocate GPU matrices
    cudaMalloc(&d_X_1, size*sizeof(float));
    cudaMalloc(&d_X_2, size*sizeof(float));
    cudaMalloc(&d_X_3, size*sizeof(float));
    cudaMalloc(&d_X_3_tmp, size*sizeof(float));
    cudaMalloc(&d_A, size*sizeof(float));
    cudaMalloc(&d_U, size*sizeof(float));

    cudaMalloc(&d_z, N*size*sizeof(float));
    cudaMalloc(&d_z_old, N*size*sizeof(float));

    cudaMalloc(&d_svd_U, m*n*sizeof(float));
    cudaMalloc(&d_svd_S, n*sizeof(float)); // size of min(m,n); cublasgesvd works for m>n
    cudaMalloc(&d_svd_S_diag, m*n*sizeof(float)); // size of min(m,n); cublasgesvd works for m>n
    cudaMalloc(&d_svd_VH, n*n*sizeof(float));
    cudaMalloc(&d_svd_temp, m*n*sizeof(float));

    // copy A to GPU
    cudaMemcpy(d_A, rpca->A, sizeof(float)*size, cudaMemcpyHostToDevice);

    // direct device allocation
    // this should be done on device kernel directly.
    thrust::device_ptr<float> dp_A(d_A);
    thrust::device_ptr<float> dp_X1(d_X_1);
    thrust::device_ptr<float> dp_X2(d_X_2);
    thrust::device_ptr<float> dp_X3(d_X_3);
    thrust::device_ptr<float> dp_X3_tmp(d_X_3_tmp);
    thrust::device_ptr<float> dp_U(d_U);
    thrust::device_ptr<float> dp_z(d_z);
    thrust::device_ptr<float> dp_z_old(d_z_old);
    thrust::device_ptr<float> dp_svd_U(d_svd_U);
    thrust::device_ptr<float> dp_svd_S(d_svd_S);
    thrust::device_ptr<float> dp_svd_S_diag(d_svd_S_diag);
    thrust::device_ptr<float> dp_svd_VH(d_svd_VH);
    thrust::device_ptr<float> dp_svd_temp(d_svd_temp);

    // initialize with zeros 
    thrust::fill(dp_X1, dp_X1 + size, fill_value);
    thrust::fill(dp_X2, dp_X2 + size, fill_value);
    thrust::fill(dp_X3, dp_X3 + size, fill_value);
    thrust::fill(dp_X3_tmp, dp_X3_tmp + size, fill_value);
    thrust::fill(dp_U, dp_U + size, fill_value);
    thrust::fill(dp_z, dp_z + size, fill_value);
    thrust::fill(dp_z_old, dp_z_old + size, fill_value);
    thrust::fill(dp_svd_S_diag, dp_svd_S_diag + n*n, fill_value);

    cublasInit();

    // local variables below for updates

    // set g2 = 0.15 * norm(A(:), inf);
    // calculate g2 as inf norm - max of sum of rows
    absolute_value<float>        abs_unary_op;
    rpca->g2 = 0.15 * thrust::transform_reduce(dp_A, dp_A + size,
                                      abs_unary_op,
                                      (float)0,
                                      thrust::maximum<float>());	

    // set g3 = 0.15 * norm(A);
    fast_svd_routine(d_A, m, n, d_svd_U, d_svd_S, d_svd_VH, 'N', 'N');
    cudaMemcpy(U, d_svd_S, sizeof(float)*n, cudaMemcpyDeviceToHost);

    rpca->g3 = 0.15 * U[0];

    printf("g2 = %f g3 = %f\n", rpca->g2, rpca->g3);

    thrust::fill(dp_U, dp_U + size, fill_value);
    // grid and block size


    printf("ADMM for RPCA is running ...\n");

    int iter;
    // GPU time
    float time = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for ( iter = 0; iter < admm_param->MAX_ITER; iter++ )
    {
        // update B here as in admm boyd - B and U are the same - we'll use U for both
        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(dp_X1, dp_X2, dp_X3, dp_A, dp_U)),
            thrust::make_zip_iterator(thrust::make_tuple(dp_X1+size, dp_X2+size, dp_X3+size, dp_A+size, dp_U+size)),
            dp_U,
            B_update(N)
        );

        // X_1 update
        thrust::transform(
            dp_X1, dp_X1 + size, dp_U, dp_X1, X1_update(admm_param->lambda)
        );

        // X_2 update
        float temp = admm_param->lambda * rpca->g2;
        thrust::transform(
            dp_X2, dp_X2 + size, dp_U, dp_X2, X2_update(temp)
        );

        // X_3 update
        // perform X_3 - B and store in X_3

        thrust::transform(
            dp_X3, dp_X3 + size, dp_U, dp_X3, thrust::minus<float>() 
        );
        fast_svd_routine(d_X_3, m, n, d_svd_U, d_svd_S, d_svd_VH, 'S', 'A');
        temp = admm_param->lambda * rpca->g3;
        thrust::transform(
            dp_svd_S, dp_svd_S + n, dp_svd_S, prox_l1(temp)
        );
        // make diagonal matrix using S
        thrust::scatter(dp_svd_S, dp_svd_S + n, thrust::make_transform_iterator(thrust::make_counting_iterator(0),diag_index(n)), dp_svd_S_diag);

        gpu_blas_mmul(d_svd_U, d_svd_S_diag, d_svd_temp, m, n, n);
        gpu_blas_mmul(d_svd_temp, d_svd_VH, d_X_3, m, n, n);
            

        // now things to do for termination checks
        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(dp_X1, dp_X2, dp_X3, dp_A)),
            thrust::make_zip_iterator(thrust::make_tuple(dp_X1+size, dp_X2+size, dp_X3+size, dp_A+size)),
            dp_svd_temp,
            avg_cal(N)
        );

    	square<float>        sq_unary_op;
        r_norm = std::sqrt(thrust::transform_reduce(
            dp_svd_temp, dp_svd_temp + size, sq_unary_op, (float)0, thrust::plus<float> ()));
        thrust::copy(dp_z, dp_z + N*size, dp_z_old);
        thrust::copy(dp_X1, dp_X1 + size, dp_z);
        thrust::copy(dp_X2, dp_X2 + size, dp_z + size);
        thrust::copy(dp_X3, dp_X3 + size, dp_z + 2*size);
        thrust::transform(dp_z, dp_z + size, dp_svd_temp, dp_z, thrust::minus<float>());
        thrust::transform(dp_z + size, dp_z + 2*size, dp_svd_temp, dp_z + size, thrust::minus<float>());
        thrust::transform(dp_z + 2*size, dp_z + 3*size, dp_svd_temp, dp_z + 2*size, thrust::minus<float>());

        thrust::transform(dp_z, dp_z + N*size, dp_z_old, dp_z_old, thrust::minus<float>());
        s_norm = std::sqrt(thrust::transform_reduce(
            dp_z_old, dp_z_old + N*size, sq_unary_op, (float)0, thrust::plus<float> ()));

        thrust::copy(dp_z, dp_z + N*size, dp_z_old);
        // eps_pri and eps_dual
        float zfro = std::sqrt(thrust::transform_reduce(
            dp_z, dp_z + N*size, sq_unary_op, (float)0, thrust::plus<float> ()));
        thrust::copy(dp_X1, dp_X1 + size, dp_z);
        thrust::copy(dp_X2, dp_X2 + size, dp_z + size);
        thrust::copy(dp_X3, dp_X3 + size, dp_z + 2*size);
        float xfro = std::sqrt(thrust::transform_reduce(
            dp_z, dp_z + N*size, sq_unary_op, (float)0, thrust::plus<float> ()));
        thrust::copy(dp_z_old, dp_z_old + N*size, dp_z);
        float eps_pri = std::sqrt(m*n*N)*admm_param->abstol + admm_param->reltol*(xfro > zfro?xfro:zfro);
        float ufro = std::sqrt(thrust::transform_reduce(
            dp_U, dp_U + size, sq_unary_op, (float)0, thrust::plus<float> ()));
        float eps_dual = std::sqrt(m*n*N)*admm_param->abstol + admm_param->reltol*std::sqrt(N)*(ufro);
        printf("%d rnorm %f eps_pri %f snorm %f eps_dual %f\n", iter, r_norm, eps_pri, s_norm, eps_dual);
        if (r_norm < eps_pri && s_norm < eps_dual)
            break;
    }

    cudaMemcpy(rpca->X1, d_X_1, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaMemcpy(rpca->X2, d_X_2, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaMemcpy(rpca->X3, d_X_3, sizeof(float)*size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("iter = %d, rnorm %f, snorm %f, Time taken : %f (sec)\n", iter, r_norm, s_norm, time/1000.0);

    if (rpca->SAVEFILE)
    {
        char filename[40];
        FILE *f;
        sprintf(filename, "X1_out.dat");
        f = fopen(filename, "wb");
        fwrite (rpca->X1,sizeof(float),size,f);
        fclose(f);

        sprintf(filename, "X2_out.dat");
        f = fopen(filename, "wb");
        fwrite (rpca->X2,sizeof(float),size,f);
        fclose(f);

        sprintf(filename, "X3_out.dat");
        f = fopen(filename, "wb");
        fwrite (rpca->X3,sizeof(float),size,f);
        fclose(f);
    }

    cudaFree(d_X_1);
    cudaFree(d_X_2);
    cudaFree(d_X_3);
    cudaFree(d_X_3_tmp);
    cudaFree(d_A);
    cudaFree(d_U);
    cudaFree(d_z);
    cudaFree(d_z_old);
    cudaFree(d_svd_U);
    cudaFree(d_svd_S);
    cudaFree(d_svd_S_diag);
    cudaFree(d_svd_VH);
    cudaFree(d_svd_temp);
    delete [] U;
}

int main(const int argc, const char **argv)
{

    RPCA_ADMM *rpca = NULL;

    rpca = (struct RPCA_ADMM *) malloc( sizeof(struct RPCA_ADMM) );

    rpca->SAVEFILE = 1;
    // we'll call our input matrix A
    rpca->A = NULL;
    rpca->X1 = NULL;
    rpca->X2 = NULL;
    rpca->X3 = NULL;

    long size;
    int Asize[2];
    unsigned int dim;

    char* str;
    if (argc > 1) {
        dim = strtol(argv[1],&str,10);
    }

    // read file
    char filename[40];
    FILE *f;
    sprintf(filename, "%dA.dat",dim);
	printf("Reading file ... %s", filename);
	f = fopen ( filename , "rb" );
    if ( f == NULL ) {
        printf("Error! Can not find input file!");
        return 0;
    }

    fread(Asize, sizeof(int), 2, f);
    rpca->m = Asize[0];
    rpca->n = Asize[1];
    // decompose input into 3 matrices
    rpca->N = 3;
    size = rpca->m * rpca->n;
    rpca->A = new float[size];
    fread(rpca->A,sizeof(float),size,f);
    fclose(f);

    rpca->X1 = new float[size];
    rpca->X2 = new float[size];
    rpca->X3 = new float[size];
    printf("Input Matrix A: rows = %d, cols = %d, total size = %lu\n", rpca->m, rpca->n, size);


    int iter_size = 1;

    ADMM_para* admm_param = NULL;

    admm_param = (struct ADMM_para *) malloc( sizeof(struct ADMM_para) );

    // default value
    admm_param->lambda = 1;
    admm_param->MAX_ITER = 200;
    admm_param->abstol = 1e-4;
    admm_param->reltol = 1e-2;

    admm_param->iter_time = new float[iter_size];

    printf("Getting GPU information .....\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);       // default device

    GPUInfo gpu_info;
    gpu_info.MAX_GRID_SIZE = prop.maxGridSize[0];
    gpu_info.MAX_BLOCK_SIZE = prop.maxThreadsPerBlock;

    /*
    // if out of GPU memory, return
    float mem = (size*5*4+(rpca->m+rpca->n)*3*4+gpu_info.MAX_GRID_SIZE*gpu_info.MAX_BLOCK_SIZE*2*4)/pow(2,30);
    float GPUmem = (long)prop.totalGlobalMem/pow(2,30);
    printf("gridDim = %d, blockDim = %d, memory required = %fGB, GPU memory = %fGB\n", gpu_info.MAX_GRID_SIZE, gpu_info.MAX_BLOCK_SIZE, mem, GPUmem );
    if ( GPUmem < mem )
    {
        printf("Not enough memory on GPU to solve the problem !\n");
        return 0;
    }
    */

    cuda_rpca_admm(rpca, admm_param, &gpu_info);

    // free stuff
    delete [] rpca->A;
    delete [] rpca->X1;
    delete [] rpca->X2;
    delete [] rpca->X3;
    delete [] admm_param->iter_time;
    free(admm_param);
    free(rpca);
}
