//#define CRTDBG_MAP_ALLOC
//#include <crtdbg.h>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace std;

int Max(int a, int b) { if (a > b) return a; return b; };
int Min(int a, int b) { if (a < b) return a; return b; };

class Polynomial
{
public:
    double *poly;
    int size;

    Polynomial(int n=0)
    {
        size=n;
        if(n==0) poly=0;
        else
        {
            poly=new double[n];
            if(poly==0) {cout<<"malloc error\n";exit(1);}
            for(int i=0;i<n;i++) poly[i]=0;
        }
    }
    Polynomial(int n, char x)
    {
        switch (x)
        {
        case 'x':                                           //xn(n)
        {
            size=n+1;
            poly=new double[size];
            if(poly==0) {cout<<"malloc error\n";exit(1);}
            for(int i=0;i<n;i++) poly[i]=0;
            poly[n]=1;
        }
            break;

        case 'c':                                            //cutFunc(n)
        {
            size=n+2;
            poly=new double[size];
            if(poly==0) {cout<<"malloc error\n";exit(1);}
            poly[0]=1;
            for(int i=1;i<n;i++) poly[i]=0;
            poly[n] = -n-1;
            poly[n+1] = n;
        }
			break;
        }
    }
	Polynomial(const double* p, int n)
	{
		size = n;
		poly = new double[n];
		if (poly == 0) { cout << "malloc error\n"; exit(1); }
		for (int i = 0; i < n; i++) poly[i] = p[i];
	}

    ~Polynomial()
    {
        if(poly) delete []poly;
    }
    Polynomial(const Polynomial& p)
    {
        size=p.size;
        if(size==0) poly=0;
        else
        {
            poly=new double[size];
            if(poly==0) {cout<<"malloc error\n";exit(1);}
            for(int i=0;i<size;i++) poly[i]=p.poly[i];
        }
    }
    void operator =(const Polynomial& p)
    {
        if(poly!=0) {cout<<"error in operator=: memory leak\n";exit(1);}
        size=p.size;
        if(size==0) poly=0;
        else
        {
            poly=new double[size];
            if(poly==0) {cout<<"malloc error\n";exit(1);}
            for(int i=0;i<size;i++) poly[i]=p.poly[i];
        }
    }

    double value(double n)
    {
        double temp=0;
		double num = n;
        if(size) temp+=poly[0];
        for(int i=1;i<size;i++)
        {
            temp+=poly[i]*num;
            num*=n;
        }
        return temp;
    }

    Polynomial integ()
    {
        Polynomial result(size+1);
        result.poly[0]=0;
        for(int i=1;i<result.size;i++)
            result.poly[i]=1.0*poly[i-1]/i;

        return result;
    }
    Polynomial deriv()
    {
		if (size < 2) { Polynomial result(1); result.poly[0] = 0; return result; }
        else
        {
            Polynomial result(size-1);
            for(int i=0;i<result.size;i++)
                result.poly[i]=poly[i+1]*(i+1);
			return result;
        }
    }

    Polynomial operator * (double n)
    {
        Polynomial result(size);
        for(int i=0;i<size;i++) result.poly[i]=n*poly[i];
        return result;
    }
    Polynomial operator * (const Polynomial& p)
    {
        Polynomial result(size+p.size-1);
        for(int i=0;i<size;i++)
            for(int j=0;j<p.size;j++)
                result.poly[i+j]+=poly[i]*p.poly[j];
        return result;
    }
    Polynomial operator + (const Polynomial& p)
    {
        Polynomial result(Max(size,p.size));
        for(int i=0;i<result.size;i++)
        {
            if(i<size & i<p.size) result.poly[i]=poly[i]+p.poly[i];
            else if(i<size) result.poly[i]=poly[i];
            else result.poly[i]=p.poly[i];
        }
        return result;
    }
    Polynomial operator + (double p)
    {
        if(size==0)
        {
            Polynomial result(1);
            result.poly[0]=p;
            return result;
        }
        else
        {
            Polynomial result(size);
            result.poly[0]+=p;
            for(int i=1;i<result.size;i++)
                result.poly[i]=poly[i];
            return result;
        }
    }
    Polynomial operator - (const Polynomial &p)
    {
        Polynomial result(Max(size,p.size));
        for(int i=0;i<result.size;i++)
        {
            if(i<size & i<p.size) result.poly[i]=poly[i]-p.poly[i];
            else if(i<size) result.poly[i]=poly[i];
            else result.poly[i]=-p.poly[i];
        }
        return result;
    }

    Polynomial operator / (int n) //div_xn
    {
        if(n>size) {cout<<"error in div_xn: n>size\n";exit(1);}
        for(int i=0;i<n;i++)
            if(poly[i]!=0) {cout<<"error in div_xn: poly[i]!=0\n";exit(1);}

        Polynomial result(size-n);
        for(int i=0;i<result.size;i++)
            result.poly[i]=poly[i+n];
        return result;
    }
};

class zernikegen
{
public:
    Polynomial **zerDic;
    int size;
    // Polynomial **G1Dic;
    // Polynomial **dG1Dic;
    // Polynomial **G2Dic;
    // Polynomial **dG2Dic;
    Polynomial **GDic;
    Polynomial **dGDic;
    Polynomial **HDic;
    Polynomial **dHDic;
    Polynomial **ZDic;
    Polynomial **dZDic;
    int ncut;

    zernikegen(int nmax, int cut)
    {
        size=nmax+1;
        /************************************************Get zernike_dict*/
        zerDic=new Polynomial* [size];
        for(int i=0;i<size;i++) zerDic[i]=new Polynomial [i+1];

        for(int n=0;n<size;n++)
            for(int l=n;l>-1;l-=2)
            {
                if(l==n) zerDic[n][l]=Polynomial(n,'x');
                else if(l==n-2)
                {
					if (n >= 2)
						zerDic[n][l] = (zerDic[n][n] * (n + 0.5)) - (zerDic[n - 2][n - 2] * (n-0.5));
                    else
                        zerDic[n][l] = zerDic[n][n]*(n+0.5);
                }
                else if(l == 0)
                {
                    int n2 = 2*n;
                    double M1 = 1.0*(n2+1)*(n2-1)/(n+l+1)/(n-l);
                    double M2 = -0.5*((2*l+1)*(2*l+1)*(n2-1) + (n2+1)*(n2-1)*(n2-3))/(n+l+1)/(n-l)/(n2-3);
                    double M3 = -1.0*(n2+1)*(n+l-1)*(n-l-2)/(n+l+1)/(n-l)/(n2-3);
                    Polynomial poly(3);poly.poly[2]=1;
                    zerDic[n][l] = (poly*M1 + M2)*zerDic[n-2][l] + (zerDic[n-4][l]*M3);
                }
                else
                {
                    double L1 =1.0*(2*n+1)/(n+l+1);
                    double L2 = -1.0*(n-l)/(n+l+1);
                    Polynomial poly(2);poly.poly[1]=1;
                    zerDic[n][l] = poly*zerDic[n-1][l-1]*L1 + (zerDic[n-2][l]*L2);
                }
            }

        /*********************************************************GetG1nl_dict*/
        // G1Dic=new Polynomial* [size];
        // for(int i=0;i<size;i++) G1Dic[i]=new Polynomial [i+1];
        // dG1Dic=new Polynomial* [size];
        // for(int i=0;i<size;i++) dG1Dic[i]=new Polynomial [i+1];

        // for(int n=0;n<size;n++)
        //     for(int l=n;l>-1;l-=2)
        //     {
        //         Polynomial tempiPoly = zerDic[n][l] * Polynomial(l+2,'x');
        //         Polynomial iPoly = tempiPoly.integ();
        //         Polynomial gPoly = iPoly/(l+1);
        //         G1Dic[n][l] = gPoly;
        //         dG1Dic[n][l] = gPoly.deriv();
        //     }

        /********************************************************GetG2nl_dict*/
        // G2Dic=new Polynomial* [size];
        // for(int i=0;i<size;i++) G2Dic[i]=new Polynomial [i+1];
        // dG2Dic=new Polynomial* [size];
        // for(int i=0;i<size;i++) dG2Dic[i]=new Polynomial [i+1];

        // for(int n=0;n<size;n++)
        //     for(int l=n;l>-1;l-=2)
        //     {
        //         Polynomial poly;
        //         if(l<=1) poly=zerDic[n][l]*Polynomial(1-l,'x');
        //         else poly = zerDic[n][l]/ (l-1);

        //         Polynomial tempipoly= poly.integ();
        //         Polynomial iPoly = tempipoly*(-1) + tempipoly.value(1);
        //         Polynomial gPoly = iPoly * Polynomial(l,'x');
        //         G2Dic[n][l] = gPoly;
        //         dG2Dic[n][l] = gPoly.deriv();
        //     }

        /********************************************************GetGnl_dict*/
        GDic=new Polynomial* [size];
        for(int i=0;i<size;i++) GDic[i]=new Polynomial [i+1];
        dGDic=new Polynomial* [size];
        for(int i=0;i<size;i++) dGDic[i]=new Polynomial [i+1];

        for(int n=0;n<size;n++)
            for(int l=n;l>-1;l-=2)
            {
                Polynomial tiPoly1 = zerDic[n][l] * (Polynomial(l+2,'x'));
                Polynomial iPoly1 = tiPoly1.integ();
                Polynomial gPoly1 = iPoly1/ (l+1);
                Polynomial ipoly2;
                if (l <= 1)
                    ipoly2 = zerDic[n][l] * Polynomial(1-l,'x');
                else
                    ipoly2 = zerDic[n][l]/ (l-1);
                Polynomial tiPoly2 = ipoly2.integ();
                Polynomial iPoly2 = tiPoly2*(-1) + tiPoly2.value(1);
                Polynomial gPoly2 = iPoly2 * Polynomial(l,'x');

                Polynomial gPoly = gPoly1 + gPoly2;
                GDic[n][l] = gPoly;
                dGDic[n][l] = gPoly.deriv();
            }

            /******************************************************GetHnl_dict*/
            ncut=cut;
            if(ncut<=1) {cout<<"error: ncut<1\n";exit(1);}

            HDic=new Polynomial* [size];
            for(int i=0;i<size;i++) HDic[i]=new Polynomial [i+1];
            dHDic=new Polynomial* [size];
            for(int i=0;i<size;i++) dHDic[i]=new Polynomial [i+1];

            Polynomial cutoff = Polynomial(ncut,'c');
            for(int n=0;n<size;n++)
                for(int l=n;l>-1;l-=2)
                {
                    Polynomial poly = GDic[n][l] * cutoff;
                    HDic[n][l] = poly;
                    dHDic[n][l] = poly.deriv();
                }

            /******************************************************GetZnl_dict*/
            ncut=cut;
            if(ncut<=1) {cout<<"error: ncut<1\n";exit(1);}

            ZDic=new Polynomial* [size];
            for(int i=0;i<size;i++) ZDic[i]=new Polynomial [i+1];
            dZDic=new Polynomial* [size];
            for(int i=0;i<size;i++) dZDic[i]=new Polynomial [i+1];

            // Polynomial cutoff = Polynomial(ncut,'c');
            for(int n=0;n<size;n++)
                for(int l=n;l>-1;l-=2)
                {
                    Polynomial poly = zerDic[n][l] * cutoff;
                    ZDic[n][l] = poly;
                    dZDic[n][l] = poly.deriv();
                }


    }

    ~zernikegen()
    {
        for(int i=0;i<size;i++)
        {
            if(zerDic[i]) delete [] zerDic[i];
            // if(G1Dic[i]) delete[] G1Dic[i];
            // if(dG1Dic[i]) delete[] dG1Dic[i];
            // if(G2Dic[i]) delete [] G2Dic[i];
            // if(dG2Dic[i]) delete [] dG2Dic[i];
            if(GDic[i]) delete [] GDic[i];
            if(dGDic[i]) delete [] dGDic[i];
            if(HDic[i]) delete [] HDic[i];
            if(dHDic[i]) delete [] dHDic[i];
            if(ZDic[i]) delete [] ZDic[i];
            if(dZDic[i]) delete [] dZDic[i];
        }

        delete [] zerDic;
        // delete[] G1Dic;
        // delete[] dG1Dic;
        // delete [] G2Dic;
        // delete [] dG2Dic;
        delete [] GDic;
        delete [] dGDic;
        delete [] HDic;
        delete [] dHDic;
        delete [] ZDic;
        delete [] dZDic;
    }
};

double integral_radial(Polynomial poly1, Polynomial  poly2)
{
	Polynomial p = poly1 * poly2 * Polynomial(2, 'x');
	Polynomial ip = p.integ();
	double I = ip.value(1) - ip.value(0);
	return I;
};

/*************************************************************Legendre coefs*/
const double p0[] = { 1 };
const double p1[] = { 0,1 };
const double p2[] = {-0.5,0,1.5 };
const double p3[] = { 0,-1.5,0,2.5 };
const double p4[] = { 0.375, 0,-3.75,0,4.375 };
const double p5[] = { 0, 1.875, 0, -8.75, 0, 7.875 };
const double p6[] = { -0.3125, 0, 6.5625, 0, -19.6875, 0, 14.4375 };

/*int main()
{
	{
		zernikegen z(10, 3);
		for (int n = 0; n < 11; n++)
			for (int l = n; l > -1; l -= 2)
				cout << "n, l= " << n << ',' << l << '\t' << z.dHDic[n][l].value(2) << endl;
				//integral_radial(z.zerDic[n][l], z.zerDic[n][l])<<'\t'<<1.0 / (2 * n + 3) << endl;
	}

	_CrtDumpMemoryLeaks();
	return 0;
}*/
