#include <cstdlib> 
#include <cmath> 
#include <ctime> 
#include <vector>
#include <iostream> 
#include <fstream>
#include <string>

using namespace std;

//#include"SmallestBall.h"
#include"GetLatticeParm.h"
#include"transData.h"
#include"wyckData.h"
#include"position.h"

#define M_PI 3.14159265358979323846

class Atom
{
public:
	const char* name;
	double radius;
	bool module;
	//position p;
	//vector<position> pointset;
	Atom() {};
	Atom(const char* n, double r , bool m)                                  // , double a = 0, double b = 0, double c = 0) :p(a, b, c)
	{
		name = n;
		radius = r;
		module = m;
		/*if(m)
		{
			Ball* ball=Findsmallball(name);
			position* center=ball.center;
			if(radius==0) radius=ball.radius;

			ifstream in(filename);
			position temp;
			while(1)
			{
				in>>temp.x>>temp.y>>temp.z;
				if(temp.y==0) break;
				pointset.push_back(temp);
			}

		}
		if((!m)&r==0) radius=covalent_radii(name);*/
	};
	Atom(const Atom &a)
	{
		name = a.name;
		radius = a.radius;
		module = a.module;
		//p = a.p;
		//for (int i = 0; i < a.pointset.size(); i++) pointset.push_back(a.pointset[i]);
	}
	void operator =(const Atom &a)
	{
		name = a.name;
		radius = a.radius;
		module = a.module;
		//p = a.p;
		//pointset.clear();
		//for (int i = 0; i < a.pointset.size(); i++) pointset.push_back(a.pointset[i]);
	}
};


class WyckPos
{
public:
	char label;
	int multiplicity;
	double wyckmatrix[12];
	vector<double> rotatematrix;
	vector<double> transmatrix;
	bool unique;
	int variables;

	WyckPos(const WyckPos& w)
	{
		label = w.label;
		multiplicity = w.multiplicity;
		for (int i = 0; i < 12; i++) wyckmatrix[i] = w.wyckmatrix[i];
		for (int i = 0; i < w.rotatematrix.size(); i++) rotatematrix.push_back(w.rotatematrix[i]);
		for (int i = 0; i < w.transmatrix.size(); i++) transmatrix.push_back(w.transmatrix[i]);
		unique = w.unique;
		variables = w.variables;
	}
	void operator =(const WyckPos& w)
	{
		label = w.label;
		multiplicity = w.multiplicity;
		for (int i = 0; i < 12; i++) wyckmatrix[i] = w.wyckmatrix[i];
		rotatematrix.clear(); transmatrix.clear();
		for (int i = 0; i < w.rotatematrix.size(); i++) rotatematrix.push_back(w.rotatematrix[i]);
		for (int i = 0; i < w.transmatrix.size(); i++) transmatrix.push_back(w.transmatrix[i]);
		unique = w.unique;
		variables = w.variables;
	}
	WyckPos(char l, int m, double* w, vector<double>* r, vector<double>* t, bool u)
	{
		label = l;
		for (int i = 0; i < 12; i++) wyckmatrix[i] = w[i];
		for (int i = 0; i < r->size(); i++) rotatematrix.push_back((*r)[i]);
		for (int i = 0; i < t->size(); i++) transmatrix.push_back((*t)[i]);
		unique = u;
		multiplicity = m;
		variables = fabs(wyckmatrix[0]) + fabs(wyckmatrix[5]) + fabs(wyckmatrix[10]);
	};

	void GetPosition(double* m, position* p, position* presult)
	{
		position temp1(m[0], m[1], m[2]);
		position temp2(m[4], m[5], m[6]);
		position temp3(m[8], m[9], m[10]);
		position temp4(m[3], m[7], m[11]);

		position temp(Dotproduct(&temp1, p), Dotproduct(&temp2, p), Dotproduct(&temp3, p));
		Padd(1, 1, &temp, &temp4, presult);
		return;
	};//m[:,:-1].dot(p)+m[:,-1]

	position GetOnePosition(void)
	{
		position randp(Rand(), Rand(), Rand());
		position temp;
		GetPosition(wyckmatrix, &randp, &temp);
		
		return Standford(&temp);
	};

	void GetAllPosition(vector<position> &positions, position p)
	{
		vector<position> tempositions;
		position temp;
		positions.clear();
		double matrix[12];
		for (int i = 0; i < rotatematrix.size() / 12; i++)
		{
			for (int j = 0; j < 12; j++) matrix[j] = rotatematrix[12 * i + j];
			GetPosition(matrix, &p, &temp);
			tempositions.push_back(temp);
		}

		for (int j = 0; j < tempositions.size(); j++)
		{
			for (int i = 0; i < transmatrix.size() / 3; i++)
			{
				position tempp(transmatrix[3 * i], transmatrix[3 * i + 1], transmatrix[3 * i + 2]);
				Padd(1, 1, &tempp, &tempositions[j], &temp);
				tempp = Standford(&temp);
				if (positions.size() == 0) positions.push_back(tempp);
				else
					for (int i = 0; i < positions.size(); i++)
					{
						if (tempp == positions[i])  break;
						if (i == positions.size() - 1) { positions.push_back(tempp); break; }
					}
				if (positions.size() == multiplicity) return;
			}
		}
		return;
	};
};

vector<WyckPos> wyckpositions;

class WyckGroup
{
public:
	int count;
	vector<WyckPos>* SimilarWyck;
	WyckGroup(const WyckGroup& w)
	{
		count = w.count;
		SimilarWyck=w.SimilarWyck;
	}
	void operator =(const WyckGroup& w)
	{
		count = w.count;
		SimilarWyck = w.SimilarWyck;
	}
	WyckGroup(vector<WyckPos>* w)
	{
		count = 0;
		SimilarWyck = w;
	}
};

class Atoms
{
public:
	Atom atom;
	int number;
	int left;
	vector<WyckGroup> wyckGroups;
	vector<int> chosenWycks;
	vector<position> positions;
	vector<position> positions_wyck;
	bool UsedMostGeneral; //UsedMostGeneral is for the second method.
	Atoms(int n, const char* name, double r, bool m) :atom (name, r, m)
	{
		number = n;
		left = n;
		UsedMostGeneral = false;
	};
	Atoms(const Atoms& a)
	{
		atom = a.atom;
		number = a.number;
		left = a.left;
		for (int i = 0; i < a.wyckGroups.size(); i++) wyckGroups.push_back(a.wyckGroups[i]);
		for (int i = 0; i < a.chosenWycks.size(); i++) chosenWycks.push_back(a.chosenWycks[i]);
		for (int i = 0; i < a.positions.size(); i++) positions.push_back(a.positions[i]);
		for (int i=0;i<a.positions_wyck.size();i++) positions_wyck.push_back(a.positions_wyck[i]);
		UsedMostGeneral = a.UsedMostGeneral;
	}
	void operator =(const Atoms& a)
	{
		atom = a.atom;
		number = a.number;
		left = a.left;
		wyckGroups.clear(); chosenWycks. clear(); positions.clear(); 
		for (int i = 0; i < a.wyckGroups.size(); i++) wyckGroups.push_back(a.wyckGroups[i]);
		for (int i = 0; i < a.chosenWycks.size(); i++) chosenWycks.push_back(a.chosenWycks[i]);
		for (int i = 0; i < a.positions.size(); i++) positions.push_back(a.positions[i]);
		for (int i=0;i<a.positions_wyck.size();i++) positions_wyck.push_back(a.positions_wyck[i]);
		UsedMostGeneral = a.UsedMostGeneral;
	}

};

bool IsUsable(WyckGroup* w, Atoms* atoms)
{
	if ((*w->SimilarWyck)[0].multiplicity > atoms->left)  return false;
	if (((*w->SimilarWyck)[0].unique == true)&(w->count >= (w->SimilarWyck)->size())) return false;

	return true;
};
class Structure
{
public:
	vector<Atoms> atoms;
	double latticeparm[9];
	double volume;
	double atomvolume;
	int spg;
	bool legal;
	double maxr;
	bool UsedMostGeneral;
	Structure() { };
	Structure(const Structure& s)
	{
		for (int i = 0; i < s.atoms.size(); i++) atoms.push_back(Atoms(s.atoms[i]));
		for (int i = 0; i < 9; i++) latticeparm[i] = s.latticeparm[i];
		volume = s.volume;
		atomvolume = s.atomvolume;
		spg = s.spg;
		legal = s.legal;
		maxr = s.maxr;
		UsedMostGeneral = s.UsedMostGeneral;
	}
	void operator =(const Structure& s)
	{
		atoms.clear();
		for (int i = 0; i < s.atoms.size(); i++) atoms.push_back(Atoms(s.atoms[i]));
		for (int i = 0; i < 9; i++) latticeparm[i] = s.latticeparm[i];
		volume = s.volume;
		atomvolume = s.atomvolume;
		spg = s.spg;
		legal = s.legal;
		maxr = s.maxr;
		UsedMostGeneral = s.UsedMostGeneral;
	}

	bool AllAtomUsed(void)
	{
		for (int i = 0; i < atoms.size(); i++)
			if (atoms[i].left > 0) return false;
		return true;
	};

	/*void AddAtoms(Atoms ats)
	{
		atoms.push_back(Atoms(ats));
	};*/

	double GetVolume(void)
	{
		double*t = latticeparm;
		return t[0] * (t[4] * t[8] - t[5] * t[7]) - t[1] * (t[3] * t[8] - t[5] * t[6]) + t[2] * (t[3] * t[7] - t[4] * t[6]);
		//determinant of 3*3 matrix "latticeparm"
	};

	bool ChooseLattice(double* latticeMins, double* latticeMaxes, double volumeMin, double volumeMax)
	{
		int attempt = 0;
		double v = 0;
		while (v > volumeMax || v < volumeMin)
		{
			GetLatticeParm(latticeparm, spg, latticeMins, latticeMaxes);
			v = GetVolume();
			attempt++;
			if (attempt > 10000) { cout << "error: failed ChooseLattice(), reconsider volumeMax and volumeMin" << endl; return false; }
		}
		volume = v;
		return true;
	};
	void ChooseWyck(void)
	{
		for (int i = 0; i < atoms[0].wyckGroups.size(); i++)
		{
			if ((*atoms[0].wyckGroups[i].SimilarWyck)[0].unique == true)
			{
				int count = 0;
				for (int j = 0; j < atoms.size(); j++)
					count += atoms[j].wyckGroups[i].count;
				vector<int> chosen;
				while (chosen.size() < count)
				{
					bool l = true;
					int temp = rand() % (atoms[0].wyckGroups[i].SimilarWyck->size());
					for (int k = 0; k < chosen.size(); k++)
					{
						if (temp == chosen[k]) {l = false; break;}	
					}
					if (l == true) chosen.push_back(temp);
				}
				count = 0;
				for (int j = 0; j < atoms.size(); j++)
				{
					for (int k = count; k < count+atoms[j].wyckGroups[i].count; k++) 
						atoms[j].chosenWycks.push_back((*atoms[0].wyckGroups[i].SimilarWyck)[chosen[k]].label-'a');
					count += atoms[j].wyckGroups[i].count;
				}
			}
			else
			{
				for (int j = 0; j < atoms.size(); j++) 
				{
					int chosen = 0;
					while (chosen < atoms[j].wyckGroups[i].count)
					{
						int k = rand() % (atoms[j].wyckGroups[i].SimilarWyck->size());
						atoms[j].chosenWycks.push_back((*atoms[0].wyckGroups[i].SimilarWyck)[k].label - 'a');
						chosen++;
					}
				}				
			}			
		}
		return;
	}

	void MakeCrystal(double threshold, int maxAttemps)
	{
		int num = 0;
		int attempt = 0;
		bool shouldadd;

		for (int i = 0; i < atoms.size(); i++)
		{
			for (int j = 0; j < atoms[i].chosenWycks.size(); j++)
			{
				attempt = 0;
				while (attempt < abs(wyckpositions[atoms[i].chosenWycks[j]].variables)*maxAttemps / 2 + 1)
				{
					shouldadd = true;
					vector<position> p;
					position pos_wyck=wyckpositions[atoms[i].chosenWycks[j]].GetOnePosition();
					wyckpositions[atoms[i].chosenWycks[j]].GetAllPosition(p, pos_wyck);
					if (p.size() != wyckpositions[atoms[i].chosenWycks[j]].multiplicity) { shouldadd = false; attempt++; continue; }
					if (CheckDistance(&p, &(atoms[i].positions), atoms[i].atom.radius, latticeparm, threshold) == false)
					{
						shouldadd = false;
						attempt++;
						continue;
					}
					for (int k = 0; k < i; k++)
					{
						if (CheckDistance(&p, &(atoms[k].positions), atoms[i].atom.radius, atoms[k].atom.radius, latticeparm, threshold) == false)
						{
							shouldadd = false;
							break;
						}
					}
					if (shouldadd)
					{
						for (int ii = 0; ii < p.size(); ii++) atoms[i].positions.push_back(p[ii]);
						atoms[i].positions_wyck.push_back(pos_wyck);
						break;
					}
					attempt++;
				}
				if (shouldadd == false)
				{
					legal = false;
					return;
				}
			}
		}
		for (int i = 0; i < atoms.size(); i++)
			if (atoms[i].positions.size() != atoms[i].number)
			{
				legal = false;
				cout << atoms[i].atom.name << '\t' << atoms[i].positions.size() << '\t' << atoms[i].number << endl;
				for (int i = 0; i < atoms.size(); i++)
				{
					cout << "errorlog for " << atoms[i].atom.name << " : ";
					for (int j = 0; j < atoms[i].chosenWycks.size(); j++)
						cout << wyckpositions[atoms[i].chosenWycks[j]].multiplicity << wyckpositions[atoms[i].chosenWycks[j]].label << ",";
					cout << endl;
				}
				return;
			}
		legal = true;
		return;
	};

	/*void newposition(position* presult, position* p0, position* p1, position* center, double theta, double phi)
	{
		position vec;
		Padd(1, -1, p1, center, &vec);
		double newvec[3];
		newvec[0] = vec.x*cos(theta) - vec.y*sin(theta)*cos(phi) + vec.z*sin(theta)*sin(phi);
		newvec[1] = vec.x*sin(theta) + vec.y*cos(theta)*cos(phi) - vec.z*cos(theta)*sin(phi);
		newvec[2] = vec.y*sin(phi) + vec.z*cos(phi);
		double temp[3][3];
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)	temp[i][j] = latticeparm[3 * i + j];
		double fvec[3];
		solve(temp, newvec, fvec);
		presult->x = p0->x + fvec[0]; presult->y = p0->y + fvec[1]; presult->z = p0->z + fvec[2];
		return;
		//vec=p1-center 
		//M1=[[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
		//M2=[[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]]
		//newvec=M1.dot(M2).dot(vec)
		//fvec=LA.solve(self.latticeparm.T,newvec) 
	};*/

	void WritePoscar(string* filename,char c)
	{
		ofstream out((*filename).c_str());
		switch (c)
		{
		case 'v':
			out << (*filename).c_str() << '\n';
			if (legal)
			{
				out << "1.0\n";
				for (int i = 0; i < 3; i++)
					out << latticeparm[i] << '\t' << latticeparm[i + 3] << '\t' << latticeparm[i + 6] << '\n';
				for (int i = 0; i < atoms.size(); i++) out << atoms[i].atom.name << '\t';
				out << '\n';
				
				for (int i = 0; i < atoms.size(); i++) out << atoms[i].number << '\t';
				out << "\nDirect\n";
				for (int i = 0; i < atoms.size(); i++)
					for (int j = 0; j < atoms[i].positions.size(); j++)
						out << atoms[i].positions[j].x << '\t' << atoms[i].positions[j].y << '\t' << atoms[i].positions[j].z << '\n';
			}
			out.close();
			break;

		case 'g':
			out << "opti conj conp\nswitch_minimiser bfgs gnorm 0.5\nvectors\n";
			if (legal)
			{
				for (int i = 0; i < 3; i++)
					out << latticeparm[i] << '\t' << latticeparm[i + 3] << '\t' << latticeparm[i + 6] << '\n';
				out << "fractional\n";
				for (int i = 0; i < atoms.size(); i++)
					for (int j = 0; j < atoms[i].positions.size(); j++)
						out << atoms[i].atom.name << " core " << atoms[i].positions[j].x << '\t' << atoms[i].positions[j].y << '\t' << atoms[i].positions[j].z << '\n';
				out << "species\nTi 2.196\nO -1.098\nbuck\n";
				out << "Ti Ti 31120.1 0.1540 5.25 15\nO  O  11782.7 0.2340 30.22 15\nTi O  16957.5 0.1940 12.59 15\n";
				out << "lennard 12 6\nTi Ti 1 0 15\nO  O  1 0 15\nTi O  1 0 15\n";
			}
			out.close();
			break;
			
		case 't':
			if(legal)
			{
				/*out<<"import spglib\nlattice=[";
				for (int i = 0; i < 3; i++)
					out << '['<<latticeparm[i] << ',' << latticeparm[i + 3] << ',' << latticeparm[i + 6] << "],\n";
				out<<"]\n";
				out<<"positions=[";
				for (int i = 0; i < atoms.size(); i++)
						for (int j = 0; j < atoms[i].positions.size(); j++)
							out << '['<<atoms[i].positions[j].x << ',' << atoms[i].positions[j].y << ',' << atoms[i].positions[j].z << "],\n";
				out<<"]\n";
				out<<"numbers=[1,]*";
				int count=0;
				for (int i = 0; i < atoms.size(); i++) count += atoms[i].number;
				out <<count<<'\n';
				out<<"cell=(lattice,positions,numbers)\n";
				out<<"lattice, scaled_positions, numbers = spglib.find_primitive(cell, symprec=1e-5)\nprint(lattice)\nprint(scaled_positions)\n";*/
				
				out<<"!useKeyWords\n!title\n"<< (*filename).c_str() << "\n!latticeTolerance\n0.0001\n";
				out << "!latticeBasisVectors" << '\n';
				for (int i = 0; i < 3; i++)
					out << latticeparm[i] << '\t' << latticeparm[i + 3] << '\t' << latticeparm[i + 6] << '\n';
				int count = 0;
				for (int i = 0; i < atoms.size(); i++) count += atoms[i].number;
				out << "!atomCount"<<'\n'<<count<<'\n'<<"!atomType"<<'\n';
				for (int i = 0; i < atoms.size(); i++) out << atoms[i].number << "*" << atoms[i].atom.name << ' ';
				out << '\n' << "!atomPosition" << '\n';
				for (int i = 0; i < atoms.size(); i++)
						for (int j = 0; j < atoms[i].positions.size(); j++)
							out << atoms[i].positions[j].x << '\t' << atoms[i].positions[j].y << '\t' << atoms[i].positions[j].z << '\n';
			}
			out.close();
			break;
		}
		return;



	};

	void AddWyckGroup(vector<WyckGroup>* wycks)
	{
		for(int i=0;i<atoms.size();i++)
			for (int j = 0; j < wycks->size(); j++)
			{
				atoms[i].wyckGroups.push_back((*wycks)[j]);
			}
		return;
	}
};

//Here begins the first solution!
void AddWyck(Structure &s, int i, vector<WyckGroup> &wycks, int j)
{
	s.atoms[i].left -= (*wycks[j].SimilarWyck)[0].multiplicity;
	wycks[j].count++;
	s.atoms[i].wyckGroups[j].count++;
	return;
};

void Displace(Structure& news, Structure old)
{
	news.legal = true;
	for (int i = 0; i < 9; i++) news.latticeparm[i] = old.latticeparm[i];
	for (int i = 0; i < old.atoms.size(); i++)
	{
		if (old.atoms[i].atom.module == false) news.atoms.push_back(old.atoms[i]);
	}


	/*for(int i=0;i<old.atoms.size();i++)
	{
		if(old.atoms[i].atom.module)
		{
			for(int j=0;j<old.atoms[i].positions.size();j++)
			{
				double theta=2*rand01()*M_PI;
				double phi=2*rand01()*M_PI;
				for(int k=0;k<old.atoms[i].atom.PointSet.size();k++)
				{
					bool Exist=false;
					Atom a=Atom(old.atoms[i].atom.name,newposition(old.atoms[i].positions[j],old.atoms[i].atom.PointSet[k],old.atoms[i].atom.p,theta,phi));////////////atoms[i].atom.p
					for(int ii=0;ii<news.atoms.size();ii++)
						if(strcmp(news.atoms[ii].atom.name,old.atoms[i].atom.name))
						{
							news.atoms[ii].AddAtom(&a);
							Exist=true;
							break;
						}
					if(!Exist)
						news.AddAtoms(&(Atoms(0,old.atoms[i].atom.name)));
						news.atoms[news.atoms.size()-1].AddAtom(&a);
				}
			}
		}
	}*/
	return;
};
bool GetAtomsCanUse(int &i, int &j, Structure* s, vector<WyckGroup>* w,vector<int>* biasedw,int wsum)
{
	for (int attempt = 0; attempt < s->atoms.size() * 5; attempt++)
	{
		i = rand() % s->atoms.size();
		if ((s->atoms)[i].left > 0)
		{
			for (int a = 0; a < w->size() *5; a++)
			{
				int r=rand()%wsum;
				int t=0;
				for(j=0;j<biasedw->size();j++)
				{
					t+=(*biasedw)[j];
					if(t>r) break;
				}
				
				if (IsUsable(&(*w)[j], &(s->atoms)[i])) return true;
			}
		}
			
	}
	for (int k = 0; k < s->atoms.size(); k++)
		if ((s->atoms)[k].left > 0)
			for (j = w->size()-1; j >=0; j--) if (IsUsable(&(*w)[j], &(s->atoms)[k]))
			{
				i = k;
				return true;
			}
	i = 0; j = 0;
	return false;
};
void GetAllCombination(Structure structure, vector<WyckGroup> wycks, vector<Structure> &combinations,bool forceMostGeneralWyckPos,vector<int>* biasedw,int wsum)
{
	if (structure.AllAtomUsed())
	{
		if (forceMostGeneralWyckPos == true)
		{
			if (wycks[wycks.size() - 1].count > 0)
				{structure.UsedMostGeneral = true;combinations.push_back(structure); }
			return;
		}			
		combinations.push_back(structure);
		return;
	}
	int i = 0, j = 0;
	if (GetAtomsCanUse(i, j, &structure, &wycks,biasedw,wsum))
	{
		AddWyck(structure, i, wycks, j);
		GetAllCombination(structure, wycks, combinations, forceMostGeneralWyckPos,biasedw,wsum);
	}
	return;
};
//The first solution ends here.
//And here begins the second solution, just for test!


void AddWyck(Atoms* atoms, WyckGroup* wyck)
{
	atoms->left -= (*wyck->SimilarWyck)[0].multiplicity;	
	wyck->count++;
	return;
}
void GetCombinationforAtoms(Atoms atoms, vector<WyckGroup> wycks, vector<Atoms> &combinations,int i)
{
	for (int j = i; j < wycks.size(); j++)
	{
		if (IsUsable(&(wycks[j]), &atoms) == true)
		{
			if(j!=wycks.size()-1) GetCombinationforAtoms(atoms, wycks, combinations, j + 1);
			AddWyck(&atoms, &(wycks[j]));
			if (atoms.left == 0)
			{
				for (int i = 0; i < wycks.size(); i++) atoms.wyckGroups[i].count=wycks[i].count;
				if (wycks[wycks.size() - 1].count > 0) atoms.UsedMostGeneral = true;
				combinations.push_back(atoms);
				return;
			}
			else GetCombinationforAtoms(atoms, wycks, combinations, j);
			break;
		}
	}
	return;
}
bool CheckUnique(Structure* s, Atoms* ats,vector<int>* Uniquewycks)
{
	for(int i=0;i<Uniquewycks->size();i++)
	{
		int count = ats->wyckGroups[(*Uniquewycks)[i]].count;
		for (int k = 0; k < s->atoms.size(); k++)
				count += s->atoms[k].wyckGroups[(*Uniquewycks)[i]].count;
		if (count > (ats->wyckGroups[(*Uniquewycks)[i]].SimilarWyck)->size())
			return false;
	}
	return true;
}

bool AddAtomstoStructure(Structure* structure, Atoms* atoms, int i, vector<Structure> &combinations, vector<int>* Uniquewycks)
{
	if (CheckUnique(structure, atoms, Uniquewycks) == true)
	{
		Structure s(*structure);
		for (int l = 0; l < s.atoms[i].wyckGroups.size(); l++)
			s.atoms[i].wyckGroups[l].count = atoms->wyckGroups[l].count;
		s.atoms[i].UsedMostGeneral = atoms->UsedMostGeneral;
		s.UsedMostGeneral = (s.UsedMostGeneral || s.atoms[i].UsedMostGeneral);
		combinations.push_back(s);
		return true;
	}
	return false;
};

void GetAllCombinations(Structure structure, vector<WyckGroup> wycks, vector<Structure> &combinations, bool forceMostGeneralWyckPos, int attemptstoGetCombs)
{
	vector<Structure> tempcombs;

	vector<int> Uniquewycks;
	for (int i = 0; i < wycks.size(); i++)
		if ((*wycks[i].SimilarWyck)[0].unique == true) Uniquewycks.push_back(i);

	vector< vector<Atoms> > combinationsofAtoms;
	for (int i = 0; i < structure.atoms.size(); i++)
	{
		vector<Atoms> comb;
		GetCombinationforAtoms(structure.atoms[i], wycks, comb, 0);
		combinationsofAtoms.push_back(comb);
		//Here begins the logfile.
		/*for (int j = 0; j < comb.size(); j++)
		{
			cout << structure.atoms[i].atom.name <<" comb "<<j<<'\n';
			for (int k = 0; k < comb[j].wyckGroups.size(); k++)
			{
				cout << comb[j].wyckGroups[k].count << "(";

				for (int l = 0; l < (comb[j].wyckGroups[k].SimilarWyck)->size(); l++)
					cout << (*comb[j].wyckGroups[k].SimilarWyck)[l].multiplicity << (*comb[j].wyckGroups[k].SimilarWyck)[l].label << ',';

				cout<<") , ";
			}
			cout << endl;
		}*/
		//And it ends here!
	}

	int combnum = 1;
	for (int i = 0; i < structure.atoms.size(); i++) combnum *= combinationsofAtoms[i].size();

	if(combnum<=1500)
	{
		tempcombs.push_back(structure);
		for (int i = 0; i < structure.atoms.size(); i++)
		{
			int n = tempcombs.size();
			for (int j = 0; j < n; j++)
			{
				for (int k = 0; k < combinationsofAtoms[i].size(); k++)
					AddAtomstoStructure(&tempcombs[j], &combinationsofAtoms[i][k], i, tempcombs, &Uniquewycks);			
			}
			tempcombs.erase(tempcombs.begin(), tempcombs.begin() + n);
		}
	}
	else	
	{
		cout << "Notice: The number of all combinations can be up to about " << combnum << ", so we just ignored some of them randomly." << endl;
		int attempt = 0;
		vector<Structure> temps;		

		while (attempt < attemptstoGetCombs)
		{
			temps.push_back(structure);
			bool l = true;
			for (int i = 0; i < structure.atoms.size(); i++)
			{
				int k = rand() % combinationsofAtoms[i].size();
				l=AddAtomstoStructure(&temps[temps.size() - 1], &combinationsofAtoms[i][k], i, temps, &Uniquewycks);
				if (l == false) break;
			}
			if(l==true) tempcombs.push_back(temps[temps.size() - 1]);
			temps.clear();
			attempt++;
		}		
	}
	
	if (forceMostGeneralWyckPos == true)
	{
		for (int i = 0; i < tempcombs.size(); i++)
		{
			if(tempcombs[i].UsedMostGeneral==true) combinations.push_back(tempcombs[i]);
		}
	}
	else
		for (int i = 0; i < tempcombs.size(); i++)
			combinations.push_back(tempcombs[i]);

	return;
}

//The second solution ends here.



void Initialize(Structure &structure, int spg, vector<Atoms> atomlist, vector<WyckPos> &wycks)
{
	wycks.clear();
	vector<double> symmetry;
	int temps = spg - 1;
	int symmetrynum = (int)trans[temps][0];
	int translatenum = (int)trans[temps][1];
	for (int i = 0; i < 12 * symmetrynum; i++)
		symmetry.push_back(trans[temps][i + 2]);
	vector<double> translate;
	for (int i = 0; i < 3 * translatenum; i++)
		translate.push_back(trans[temps][12 * symmetrynum + i + 2]);
	int wycksnum = (int)wyck[temps][0];
	for (int i = 0; i < wycksnum; i++)
	{
		char label = (char)wyck[temps][15 * i + 1];
		double matrix[12];
		for (int j = 0; j < 12; j++)
			matrix[j] = wyck[temps][15 * i + j + 2];
		int multiplicity = (int)wyck[temps][15 * i + 14];
		bool unique = (int)wyck[temps][15 * i + 15];
		wycks.push_back(WyckPos(label, multiplicity, matrix, &symmetry, &translate, unique));
	}

	double atomvolume = 0;
	double maxr = 0;

	for (int i = 0; i < atomlist.size(); i++)
	{
		structure.atoms.push_back(atomlist[i]);
		atomvolume += 4 * M_PI / 3 * atomlist[i].number*pow((atomlist[i].atom.radius), 3);
		maxr = Max(maxr, atomlist[i].atom.radius);
	}

	structure.atomvolume = atomvolume;
	structure.spg = spg;
	structure.maxr = maxr;
	structure.UsedMostGeneral = false;

	return;
}
void GetWyckPosGrouped(vector<WyckGroup>& wycks,int i)
{
	vector<WyckPos>* SimilarWyck = new vector<WyckPos>;
	SimilarWyck->push_back(wyckpositions[i]);
	wycks.push_back(WyckGroup(SimilarWyck));
	if (i == wyckpositions.size() - 1) return;
	
	int j;
	for (j = i + 1; j < wyckpositions.size(); j++)
	{
		if ((wyckpositions[j].multiplicity == wyckpositions[i].multiplicity)&(wyckpositions[j].unique == wyckpositions[i].unique))
			SimilarWyck->push_back(wyckpositions[j]);
		else break;
	}
	GetWyckPosGrouped(wycks, j);
};
void GetWycksDeleted(vector<WyckGroup>& wycks)
{
	for (int i = 0; i < wycks.size(); i++)
	{
		wycks[i].SimilarWyck->clear();
		delete wycks[i].SimilarWyck;
	}
	return;
};