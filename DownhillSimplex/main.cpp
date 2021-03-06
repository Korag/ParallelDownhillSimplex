// DownhillSimplex.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "pch.h"
#include "DownhillSimplex.h"
#include <iostream>
#include <mpi.h>
#include <time.h>
//#include "windows.h"

#include <chrono>
#include <thread>

using namespace std;

// Example using amoeba
//
// Function to be minimized is
// x^2 - 4x + y^2 - y - xy
//
// The function argument is a vector whose size is 2.
// z[0] corresponds to x and z[1] corresponds to y


Doub func(VecDoub_I & z)
{
	//std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return SQR(z[0]) - 4.0*z[0] + SQR(z[1]) - z[1] - z[0] * z[1];
}

int main(int argc, char* argv[])
{
    const Doub FTOL = 1.0e-16;

    const Int mp = 3;
    const Int np = 2;

    Doub p_array[mp*np] = {
        0.0, 0.0,
        1.2, 0.0,
        0.0, 0.8,
    };

    MatDoub p(mp, np, p_array);
	
	// MPI
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

		clock_t start, stop;
		double timeElapsed;

		if (rank == master)
		{
			cout << fixed << setprecision(6);

			cout << "Vertices of initial simplex:" << endl << endl;
			cout << setw(3) << "i" << setw(10) << "x[i]" << setw(12) << "y[i]" << endl;
			cout << " --------------------------" << endl;
			for (Int i = 0; i < mp; i++) {
				cout << setw(3) << i;
				for (Int j = 0; j < np; j++) {
					cout << setw(12) << p[i][j];
				}
				cout << endl;
			}
			cout << endl;
		}

		DownhillSimplex a(FTOL);
		start = clock();

		a.minimize(p, func, rank, size, status);
		
		
		stop = clock();

		if (rank == master)
		{
			timeElapsed = (double)(stop - start) / CLOCKS_PER_SEC;
			cout << "Czas wykonania: " << timeElapsed << endl;

			cout << "Number of function evaluations: " << a.nfunc << endl << endl;
			cout << "Vertices of final simplex and function" << endl;
			cout << "values at the vertices:" << endl << endl;
			cout << setw(3) << "i" << setw(10) << "x[i]" << setw(12) << "y[i]";
			cout << setw(14) << "function" << endl;
			cout << " --------------------------------------" << endl;
			for (Int i = 0; i < mp; i++) {
				cout << setw(3) << i;
				for (Int j = 0; j < np; j++) {
					cout << setw(12) << a.p[i][j];
				}
				cout << setw(12) << a.y[i] << endl;
			}
			cout << endl;
		}

	system("pause");
	return 0;
}