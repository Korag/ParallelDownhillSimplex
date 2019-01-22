#include <vector>
#include "nr3.h"
#include <mpi.h>

#define master 0

using namespace std;

struct DownhillSimplex {
	const Doub ftol;
	Int nfunc;
	Int mpts;
	Int ndim;
	Doub fmin;
	VecDoub y;
	MatDoub p;
	DownhillSimplex(const Doub ftoll) : ftol(ftoll) {}
	template <class T>
	VecDoub minimize(VecDoub_I &point, const Doub del, T &func)
	{
		VecDoub dels(point.size(),del);
		return minimize(point,dels,func);
	}
	template <class T>
	VecDoub minimize(VecDoub_I &point, VecDoub_I &dels, T &func)
	{
		Int ndim=point.size();
		MatDoub pp(ndim+1,ndim);
		for (Int i=0;i<ndim+1;i++) {
			for (Int j=0;j<ndim;j++)
				pp[i][j]=point[j];
			if (i !=0 ) pp[i][i-1] += dels[i-1];
		}
		return minimize(pp,func);
	}
	template <class T>
	VecDoub minimize(MatDoub_I &pp, T &func, int &rank, int &size, MPI_Status &status)
	{
		// Zmienne pomocnicze
		double ytrS;
		double yINHI;
		Doub ytry;

		const Int NMAX=5000;
		const Doub TINY=1.0e-10;
		int ihi,ilo,inhi;
		mpts=pp.nrows();
		ndim=pp.ncols();
		VecDoub psum(ndim),pmin(ndim),x(ndim);
		p=pp;

		y.resize(mpts);
		for (Int i=0;i<mpts;i++)
		{
			
			if (rank == i+1)
			{
				for (Int j = 0; j<ndim; j++)
					x[j] = p[i][j];
					y[i] = func(x);
					MPI_Send(&y[i], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
					// A¿ wszystkie w¹tki wyœl¹
					MPI_Barrier(MPI_COMM_WORLD);
					break;
			}
		}
		if (rank == master)
		{
			// A¿ wszystkie w¹tki wyœl¹
			MPI_Barrier(MPI_COMM_WORLD);
			for (int i = 0; i < size - 1; i++)
			{
				MPI_Recv(&y[i], 1, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD, &status);
			};
			nfunc = 0;
			get_psum(p, psum);
		}
		
		for (;;) {
			if (rank == master)
			{			
				ilo=0;
				ihi = y[0]>y[1] ? (inhi=1,0) : (inhi=0,1);
				for (Int i=0;i<mpts;i++) {
					if (y[i] <= y[ilo]) ilo=i;
					if (y[i] > y[ihi]) {
						inhi=ihi;
						ihi=i;
					} else if (y[i] > y[inhi] && i != ihi) inhi=i;
				}
				Doub rtol=2.0*abs(y[ihi]-y[ilo])/(abs(y[ihi])+abs(y[ilo])+TINY);
				if (rtol < ftol) {
					SWAP(y[0],y[ilo]);
					for (Int i=0;i<ndim;i++) {
						SWAP(p[0][i],p[ilo][i]);
						pmin[i]=p[0][i];
					}
					fmin=y[0];
					return pmin;
				}
				if (nfunc >= NMAX) throw("NMAX exceeded");
				nfunc += 2;
				
				ytry=amotry(p,y,psum,ihi,-1.0,func);
				if (ytry <= y[ilo])
					ytry=amotry(p,y,psum,ihi,2.0,func);
				
				ytrS = (double)ytry;
				for (int i = 1; i < size; i++)
				{
					// wyslac do slavow ytr, y[inhi] zeby weszly do bloku else if
					MPI_Send(&ytrS, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
					MPI_Send(&inhi, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
					// wyslac ilo zeby weszly do ifa w nastepnej petli
					MPI_Send(&ilo, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
					// wyslac wartosc y[inhi] bo slavy nie maja pelnego wektora y
					double yINHI = y[inhi];
					MPI_Send(&yINHI, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
					MPI_Send(&ihi, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				}
				MPI_Barrier(MPI_COMM_WORLD);
			}

			// Slavy czekaj¹ na mastera
			if (rank != master)
			{
				MPI_Recv(&ytrS, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&inhi, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&ilo, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&yINHI, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&ihi, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
				ytry = (Doub)ytrS;
				y[inhi] = yINHI;
				// Slavy odbieraja dane wys³ane przez mastera
				MPI_Barrier(MPI_COMM_WORLD);
			}
		
			if (ytry >= y[inhi]) 
			{		
				// Tutaj nie wchodza slavy a powinny maja te wartosci z else ifa --> odtad trzeba naprawic
				/*if (rank != master)
				{
					cout << "Jestê slejwe³e: " << rank << endl;
				}
				if (rank == master)
				{
					cout << "Jestem masterê" << endl;
				}*/
				Doub ysave=y[ihi];
				ytry=amotry(p,y,psum,ihi,0.5,func);
				if (ytry >= ysave)
				{
					for (Int i=0;i<mpts;i++) 
					{
						if (rank == i + 1)
						{
							if (i != ilo)
							{
								for (Int j = 0; j < ndim; j++)
									p[i][j] = psum[j] = 0.5*(p[i][j] + p[ilo][j]);

								y[i] = func(psum);

								// wysy³anie do mastera y[i]
								MPI_Send(&y[i], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
								// A¿ wszystkie w¹tki wyœl¹
								MPI_Barrier(MPI_COMM_WORLD);
								break;
							}
						}
					}

					// odbieranie przez mastera wszystkich y[i]
					if (rank == master)
					{
						// A¿ wszystkie w¹tki wyœl¹
						MPI_Barrier(MPI_COMM_WORLD);
						for (int i = 0; i < size - 1; i++)
						{
							MPI_Recv(&y[i], 1, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD, &status);
						}
					}
					nfunc += ndim;
					get_psum(p, psum);
					MPI_Barrier(MPI_COMM_WORLD);
				}
			}
			else --nfunc;
		}
	}
	inline void get_psum(MatDoub_I &p, VecDoub_O &psum)
	{
		for (Int j=0;j<ndim;j++) {
			Doub sum=0.0;
			for (Int i=0;i<mpts;i++)
				sum += p[i][j];
			psum[j]=sum;
		}
	}
	template <class T>
	Doub amotry(MatDoub_IO &p, VecDoub_O &y, VecDoub_IO &psum,
		const Int ihi, const Doub fac, T &func)
	{
		VecDoub ptry(ndim);
		Doub fac1=(1.0-fac)/ndim;
		Doub fac2=fac1-fac;
		for (Int j=0;j<ndim;j++)
			ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;
		Doub ytry=func(ptry);
		if (ytry < y[ihi]) {
			y[ihi]=ytry;
			for (Int j=0;j<ndim;j++) {
				psum[j] += ptry[j]-p[ihi][j];
				p[ihi][j]=ptry[j];
			}
		}
		return ytry;
	}
};
