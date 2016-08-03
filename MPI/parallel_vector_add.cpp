/*
	@author Ken O'Brien <kennetho@xilinx.com>
	
	Parallel vector addition using synchronous MPI.
	
	Uses one master and three slave processes. One slave per addition operation. For illustrative purposes only.
	
	To run:
 		mpirun -np 4 ./parallel_vector_add
*/

#include <iostream>
#include <stdexcept>
#include <mpi.h>

constexpr size_t N = 3;
using Vector = float[N];


void checkError(int err) {
	if(err != MPI_SUCCESS) {
		int err_length = MPI_MAX_ERROR_STRING;
		char err_buffer[err_length];
		MPI_Error_string(err, err_buffer, &err_length);
		throw std::domain_error("MPI ERROR: "+ std::string(err_buffer));
	}
}

int main(int argc, char *argv[]) {
	checkError(MPI_Init(&argc, &argv));
	
	int rank, size;

	checkError(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	checkError(MPI_Comm_size(MPI_COMM_WORLD, &size));

	Vector c;
	MPI_Status status;
	if(rank == 0) { // Master
		Vector a = {1, 2, 3};
		Vector b = {5, 6, 8};
		for(int i=1; i<=N; ++i) {
			checkError(MPI_Send(&a[i-1], 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD));
			checkError(MPI_Send(&b[i-1], 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD));
			
			checkError(MPI_Recv(&c[i-1], 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status));
		}
	} else { // Slave
		float buf_a, buf_b;
		checkError(MPI_Recv(&buf_a, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status));
		checkError(MPI_Recv(&buf_b, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status));
		buf_a += buf_b;
		checkError(MPI_Send(&buf_a, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD));
	}
	checkError(MPI_Barrier(MPI_COMM_WORLD));	

	if(rank == 0) {
		std::cout << std::endl << "Result: " << std::endl;
		for(auto e: c) 
			std::cout << e << " ";
		std::cout << std::endl;
	}

	checkError(MPI_Finalize());	
	return 0;
}
