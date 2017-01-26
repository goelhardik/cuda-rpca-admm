GENERATING MATRIX
# Use the generate_matrix.m file to generate a random matrix of a specified size and a particular rank
# Command-
generate_matrix(25344, 200, 1);
# Here 25344 corresponds to the image size (144 x 176) and 200 corresponds to the number of frames in the video
# Note that the first dimension should be greater than the second one in generate_matrix() for our algorithm to work. (25344 > 200)

# This command will generate a 200A.dat file, which will serve as input to our RPCA algorithm


RUNNING RPCA USING ADMM USING MATLAB
# From MATLAB, use the admm_example.m file to run ADMM for RPCA. Provide the matrix generated in the previous step as input.
# Command-
admm_example('200A.dat');

# This command will run admm and write the output matrices into three different files named like boyd_X1.dat etc..

RUNNING RPCA USING ADMM USING CUDA
# For CUDA code, use the script compile_and_run.sh to compile the code and run the file. Provide the input matrix as an argument to this script.
# Command-
./compile_and_run.sh 200A.dat

# This command will run the CUDA code for RPCA and write output to files named like X1_out.dat, etc..


Other files include MATLAB files to convert output into videos.
