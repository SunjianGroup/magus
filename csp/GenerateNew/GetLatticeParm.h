#pragma once
#include <cstdlib> 
#include <cmath> 
#include <iostream> 
#define M_PI 3.14159265358979323846
using namespace std;

double Max(double a, double b)
{
	if (a > b) return a;
	else return b;
}
double Max(double a, double b, double c)
{
	if (a > b)
	{
		if (a > c) return a;
		return c;
	}
	if (b > c) return b;
	return c;
}

double Min(double a, double b)
{
	if (a < b) return a;
	return b;
}
double Min(double a, double b, double c)
{
	if (a < b)
	{
		if (a < c) return a;
		return c;
	}
	if (b < c) return b;
	return c;
}
double Rand(void) //generate numbers between 0 and 1
{
	return 1.0*rand() / RAND_MAX;
}
void GetLatticeParm(double* M, int spg, double* latticeMins, double* latticeMaxes)
{
	double latticeparm[6];
	double maxes[6], mins[6];
	int i = 0;
	for (i = 0; i < 6; i++) //copy from latticeMins/Maxes to mins/maxes
	{
		maxes[i] = *(latticeMaxes + i);
		mins[i] = *(latticeMins + i);
	}

	if ((spg >= 1) & (spg <= 2)) //ChooseTriclinic	
		for (i = 0; i < 6; i++) latticeparm[i] = Rand()*(maxes[i] - mins[i]) + mins[i];
	//a1!=a2!=a3;alpha!=beta!=gamma

	else if ((spg >= 3) & (spg <= 15)) //ChooseMonoclinic
	{
		for (i = 0; i < 6; i++) latticeparm[i] = Rand()*(maxes[i] - mins[i]) + mins[i];
		latticeparm[3] = M_PI / 2;
		latticeparm[5] = M_PI / 2;
		//a1!=a2!=a3;alpha=gamma=pi/2,!=beta
	}
	else if ((spg >= 16) & (spg <= 74)) //ChooseOrthorhombic
	{
		for (i = 0; i < 3; i++) latticeparm[i] = Rand()*(maxes[i] - mins[i]) + mins[i];
		for (i = 3; i < 6; i++) latticeparm[i] = M_PI / 2;
		//a1!=a2!=a3;alpha=beta=gamma=pi/2
	}
	else if ((spg >= 75) & (spg <= 142)) //ChooseTetragonal
	{
		double t1 = Max(mins[0], mins[1]);
		double t2 = Min(maxes[0], maxes[1]);
		latticeparm[0] = Rand()*(t2 - t1) + t1;
		latticeparm[1] = latticeparm[0];
		latticeparm[2] = Rand()*(maxes[2] - mins[2]) + mins[2];
		for (i = 3; i < 6; i++) latticeparm[i] = M_PI / 2;
		//a1=a2!=a3;alpha=beta=gamma=pi/2
	}
	else if ((spg >= 143) & (spg <= 167)) //ChooseTrigonal
	{
		double t1 = Max(mins[0], mins[1]);
		double t2 = Min(maxes[0], maxes[1]);
		latticeparm[0] = Rand()*(t2 - t1) + t1;
		latticeparm[1] = latticeparm[0];
		latticeparm[2] = Rand()*(maxes[2] - mins[2]) + mins[2];
		latticeparm[3] = M_PI / 2;
		latticeparm[4] = M_PI / 2;
		latticeparm[5] = 2 * M_PI / 3;
		//a1=a2!=a3;alpha=beta=pi/2,gamma=2pi/3	
	}
	else if ((spg >= 168) & (spg <= 194)) //ChooseHexagonal
	{
		double t1 = Max(mins[0], mins[1]);
		double t2 = Min(maxes[0], maxes[1]);
		latticeparm[0] = Rand()*(t2 - t1) + t1;
		latticeparm[1] = latticeparm[0];
		latticeparm[2] = Rand()*(maxes[2] - mins[2]) + mins[2];
		latticeparm[3] = M_PI / 2;
		latticeparm[4] = M_PI / 2;
		latticeparm[5] = 2 * M_PI / 3;
		//a1=a2!=a3;alpha=beta=pi/2,gamma=2pi/3	
	}
	else if ((spg >= 195) & (spg <= 230)) //ChooseCubic
	{
		double t1 = Max(mins[0], mins[1], mins[2]);
		double t2 = Min(maxes[0], maxes[1], maxes[2]);
		latticeparm[0] = Rand()*(t2 - t1) + t1;
		latticeparm[1] = latticeparm[0];
		latticeparm[2] = latticeparm[0];
		for (i = 3; i < 6; i++) latticeparm[i] = M_PI / 2;
		//a1=a2=a3;alpha=beta=gamma=pi/2
	}

	for (i = 0; i < 9; i++) M[i] = 0;
	M[0] = latticeparm[0]; //ax
	M[1] = latticeparm[1] * cos(latticeparm[5]);//bx
	M[4] = latticeparm[1] * sin(latticeparm[5]);//by
	M[2] = latticeparm[2] * cos(latticeparm[4]);//cx
	M[5] = (latticeparm[2] * latticeparm[1] * cos(latticeparm[3]) - M[2] * M[1]) / M[4];//cy
	M[8] = sqrt(pow(latticeparm[2], 2) - M[2] * M[2] - M[5] * M[5]);//cz
	//M=[[ax,bx,cx],[0,by,cy],[0,0,cz]]  
	for (int i = 0; i < 9; i++) if (fabs(M[i]) < 1e-6) M[i] = 0;
	return;
}


//notes here:
//alpha=b^c, beta=a^c, gamma=a^b
