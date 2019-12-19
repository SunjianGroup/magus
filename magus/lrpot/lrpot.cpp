//#include "pch.h"
#include"zernikegen.cpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
using namespace std;

#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
#include <boost/python/numpy.hpp>
namespace p = boost::python;
namespace np = boost::python::numpy;


Polynomial Legendre(int l)
{
	switch (l)
	{
	case 0:
		return Polynomial(p0, 1); break;
	case 1:
		return Polynomial(p1, 2); break;
	case 2:
		return Polynomial(p2, 3); break;
	case 3:
		return Polynomial(p3, 4); break;
	case 4:
		return Polynomial(p4, 5); break;
	case 5:
		return Polynomial(p5, 6); break;
	case 6:
		return Polynomial(p6, 7); break;
	default:
	{
		Polynomial Pn_1 = Legendre(l - 1);
		Polynomial Pn_2 = Legendre(l - 2);
		Polynomial result(l + 1);
		result.poly[0] = -1.0 * (l - 1) / l * Pn_2.poly[0];
		for (int i = 1; i <= l - 2; i++) result.poly[i] = 1.0 * (2 * l - 1) / l * Pn_1.poly[i - 1] - 1.0 * (l - 1) / l * Pn_2.poly[i];
		for (int i = l - 1; i < l + 1; i++) result.poly[i] = 1.0 * (2 * l - 1) / l * Pn_1.poly[i - 1];
		return result;
	}
	break;
	}
};

class subscripts
{
public:
	int n1; int n2; int l;
	subscripts(int a, int b, int c) { n1 = a; n2 = b; l = c; }
};
class Neighbor
{
public:
	int i; int j;
	double D; double SclD;
	double Vec[3]; double UVec[3];
	int Ele;
	vector<double> H1, H2, dH1, dH2;

	Neighbor() {};
	Neighbor(double I, double J, double d, double scalD, double v0, double v1, double v2,double ele )
	{
		i = (int) I; j = (int)J;
		D = d; SclD = scalD;
		Vec[0] = v0; Vec[1] = v1, Vec[2] = v2;
		for (int i = 0; i < 3; i++) UVec[i] = Vec[i] / D;
		Ele = (int) ele;
	}
	void Complete(double h1, double h2, double dh1, double dh2)
	{
		H1.push_back(h1); H2.push_back(h2);
		dH1.push_back(dh1); dH2.push_back(dh2);
	}
};

class CalculateFingerprints_part
{
public:
	double cutoff;
	int nmax;
	int lmax;
	vector<int> eleParm;
	zernikegen z;
	Neighbor* neighbors;
	int neighbors_size;
	vector<subscripts> subs;
	vector<double> consts;
	vector<Polynomial> angF;
	vector<Polynomial> angdF;

	int Nd;
	double* eFp;
	double**** fFps;

	int totNd, Nat;

	CalculateFingerprints_part(double Cutoff, int Nmax, int Lmax, int Ncut, bool diag = false) : z(Nmax, Ncut)
	{
		cutoff = Cutoff;
		nmax = Nmax;
		lmax = Lmax;

		/**********************************************************Get subs*/
		if (diag)
		{
			for (int n = 0; n <= nmax; n++)
				for (int l = Min(n, lmax - (n - lmax) % 2); l > -1; l -= 2)
					subs.push_back(subscripts(n, n, l));
		}
		else
		{
			for (int n1 = 0; n1 < nmax + 1; n1 += 2)
				for (int n2 = n1; n2 < nmax + 1; n2 += 2)
					for (int l = Min(n1, lmax - (n1 - lmax) % 2); l > -1; l -= 2)
						subs.push_back(subscripts(n1, n2, l));

			for (int n1 = 1; n1 < nmax + 1; n1 += 2)
				for (int n2 = n1; n2 < nmax + 1; n2 += 2)
					for (int l = Min(n1, lmax - (n1 - lmax) % 2); l > -1; l -= 2)
						subs.push_back(subscripts(n1, n2, l));
		}
		Nd = subs.size();

		/*****************************************************Get consts, angF,angdF,fpIndex*/
		for (int index = 0; index < subs.size(); index++)
		{
			int n1 = subs[index].n1; int n2 = subs[index].n2; int l = subs[index].l;
			double co = sqrt(1.0 * (2 * n1 + 3) * (2 * n2 + 3) / (2 * l + 1)) / (2 * l + 1);
			if (n1 == n2) co *= 0.5;
			consts.push_back(co);

			Polynomial angf = Legendre(l);
			angF.push_back(angf);
			angdF.push_back ( angf.deriv());
		}

		Py_Initialize();
		np::initialize();
	}

	void SeteleParm(np::ndarray eleparm)
	{
		double* temp = reinterpret_cast<double*> (eleparm.get_data());
		int len = eleparm.shape(0);
		for (int i = 0; i < len; i++)
			eleParm.push_back((int)temp[i]);
		return;
	}
	void SetNeighbors(np::ndarray neighbor_list)
	{
		int len = neighbor_list.shape(0)/7;
		neighbors= new Neighbor[len];
		neighbors_size = len;
		double* temp= reinterpret_cast<double*> (neighbor_list.get_data());
		for (int i = 0; i < len; i++)
		{
			int t = 7 * i;
			neighbors[i] = Neighbor(temp[t], temp[t + 1], temp[t + 2], temp[t+2]/cutoff, temp[t + 3], temp[t + 4], temp[t + 5], temp[t + 6]);
		}
		return;
	}

	void CompleteNeighbors()
	{
		for(int j=0;j<neighbors_size;j++)
			for (int index = 0; index < subs.size(); index++)
			{
				int n1 = subs[index].n1; int n2 = subs[index].n2; int l = subs[index].l;
				double H1j = z.ZDic[n1][l].value(neighbors[j].SclD);
				double H2j = z.ZDic[n2][l].value(neighbors[j].SclD);
				double dH1j = z.dZDic[n1][l].value(neighbors[j].SclD);
				double dH2j = z.dZDic[n2][l].value(neighbors[j].SclD);
				// double H1j = z.HDic[n1][l].value(neighbors[j].SclD);
				// double H2j = z.HDic[n2][l].value(neighbors[j].SclD);
				// double dH1j = z.dHDic[n1][l].value(neighbors[j].SclD);
				// double dH2j = z.dHDic[n2][l].value(neighbors[j].SclD);
				neighbors[j].Complete(H1j, H2j, dH1j, dH2j);
			}
		return;
	}

	void get_fingerprints(int i, int cenEleInd)
	{
		CompleteNeighbors();
		vector<int> fpindex;
		for (int index = 0; index < subs.size(); index++)
		{
			int fpIndex = cenEleInd * Nd + index;
			fpindex.push_back(fpIndex);
		}

		/**********eFp = np.zeros((totNd)); fFps = np.zeros((Nat, Nat, 3, totNd))*/
		eFp = new double[totNd];
		for (int i = 0; i < totNd; i++) eFp[i] =0;
		fFps= new double*** [Nat];
		for (int l = 0;l < Nat; l++)
		{
			fFps[l] = new double** [Nat];
			for (int i = 0; i < Nat; i++)
			{
				fFps[l][i] = new double* [3];
				for (int j = 0; j < 3; j++)
				{
					fFps[l][i][j] = new double[totNd];
					for (int k = 0; k < totNd; k++) fFps[l][i][j][k] = 0;
				}
			}
		}

		for(int j=0;j<neighbors_size;j++)
			for (int k = j + 1; k < neighbors_size; k++)
			{
				double cosjk = 0;
				for (int i = 0; i < 3; i++) cosjk += neighbors[j].UVec[i] * neighbors[k].UVec[i];

				double Djk = 0;
				for (int i = 0; i < 3; i++) Djk += pow(neighbors[j].Vec[i] - neighbors[k].Vec[i], 2);
				Djk = sqrt(Djk);

				double jkUVec[3];
				for (int i = 0; i < 3; i++) jkUVec[i] = (neighbors[k].Vec[i] - neighbors[j].Vec[i]) / Djk;

				int rho = eleParm[neighbors[j].Ele] * eleParm[neighbors[k].Ele];

				for (int index = 0; index < subs.size(); index++)
				{
					double radPart = neighbors[j].H1[index] * neighbors[k].H2[index] + neighbors[k].H1[index] * neighbors[j].H2[index];
					radPart *= rho;
					double c = consts[index];
					radPart *= c;

					double angPart = angF[index].value(cosjk);
					double dAngPart = angdF[index].value(cosjk);
					double fpVal = radPart * angPart;
					int fpIndex = fpindex[index];
					eFp[fpIndex] += fpVal;

					double dcosdij = 1.0 / neighbors[k].D - 0.5 * cosjk / neighbors[j].D;
					double dcosdik = 1.0 / neighbors[j].D - 0.5 * cosjk / neighbors[k].D;
					double dcosdjk = -1.0 * Djk / neighbors[j].D / neighbors[k].D;

					double derij = rho * c * (neighbors[j].dH1[index] * neighbors[k].H2[index] + neighbors[j].dH2[index] * neighbors[k].H1[index]) / cutoff * angPart + radPart * dAngPart * dcosdij;
					double derik = rho * c * (neighbors[k].dH1[index] * neighbors[j].H2[index] + neighbors[k].dH2[index] * neighbors[j].H1[index]) / cutoff * angPart + radPart * dAngPart * dcosdik;
					double derjk = radPart * dAngPart * dcosdjk;

					for (int a = 0; a < 3; a++)
					{
						double temp= derij * neighbors[j].UVec[a];
						fFps[i][neighbors[j].j][a][fpIndex] += temp;
						fFps[neighbors[j].j][i][a][fpIndex] -= temp;

						temp = derik * neighbors[k].UVec[a];
						fFps[i][neighbors[k].j][a][fpIndex] += temp;
						fFps[neighbors[k].j][i][a][fpIndex] -= temp;

						temp = derjk * jkUVec[a];
						fFps[neighbors[j].j][neighbors[k].j][a][fpIndex] += temp ;
						fFps[neighbors[k].j][neighbors[j].j][a][fpIndex] -= temp;
					}
				}
			}

		for(int j=0;j<neighbors_size;j++)
			for (int index = 0; index < subs.size(); index++)
			{
				double rho = eleParm[neighbors[j].Ele] * eleParm[neighbors[j].Ele];
				double	radPart = neighbors[j].H1[index] * neighbors[j].H2[index];
				radPart *= rho;
				double c = consts[index];
				radPart *= c;

				int fpIndex = fpindex[index];
				eFp[fpIndex] += radPart;
				double	derij = rho * c * (neighbors[j].dH1[index] * neighbors[j].H2[index] + neighbors[j].dH2[index] * neighbors[j].H1[index]) / cutoff;
				for (int a = 0; a < 3; a++)
				{
					double temp =  derij *neighbors[j].UVec[a];
					fFps[i][neighbors[j].j][a][fpIndex] += temp;
					fFps[neighbors[j].j][i][a][fpIndex] -= temp;
				}
			}

		delete[] neighbors;
		return;
	}

	p::list GeteFp()
	{
		p::list l;
		for (int i = 0; i < totNd; i++) l.append(eFp[i]);
		delete [] eFp;
		return l;
	}
	np::ndarray GetfFps()
	{
		p::tuple shape = p::make_tuple(Nat, Nat, 3, totNd);
		np::dtype dtype = np::dtype::get_builtin<double>();
		np::ndarray l=np::empty(shape, dtype);
		for (int n = 0; n < Nat; n++)
			for (int i = 0; i < Nat; i++)
				for (int j = 0; j < 3; j++)
					for (int k = 0; k < totNd; k++) l[n][i][j][k]=fFps[n][i][j][k];

		for (int n = 0; n < Nat; n++)
		{
			for (int i = 0; i < Nat; i++)
			{
				for (int j = 0; j < 3; j++) delete[] fFps[n][i][j];
				delete[] fFps[n][i];
			}
			delete[] fFps[n];
		}
		delete[] fFps;
		return l;
	}
};

BOOST_PYTHON_MODULE(lrpot)
{
	using namespace boost::python;
	class_<CalculateFingerprints_part>("CalculateFingerprints_part", init<double, int, int, int, bool>())
		.def_readwrite("Nd", &CalculateFingerprints_part::Nd)
		.def_readwrite("totNd", &CalculateFingerprints_part::totNd)
		.def_readwrite("Nat", &CalculateFingerprints_part::Nat)

		.def("SeteleParm", &CalculateFingerprints_part::SeteleParm)
		.def("SetNeighbors", &CalculateFingerprints_part::SetNeighbors)
		.def("get_fingerprints", &CalculateFingerprints_part::get_fingerprints)
		.def("GeteFp", &CalculateFingerprints_part::GeteFp)
		.def("GetfFps", &CalculateFingerprints_part::GetfFps)
		;
};

