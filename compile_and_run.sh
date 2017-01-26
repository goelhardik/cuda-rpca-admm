rm rpca_admm
 nvcc -o rpca_admm rpca_admm.cu rpca_admm_kernel.cu -lcublas -lcusolver
./rpca_admm $1 
