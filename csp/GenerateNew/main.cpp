#include"generatenew.cpp"

#include"vectorData.h"

#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
#include <boost/python/numpy.hpp>
namespace p = boost::python;
namespace np = boost::python::numpy;

int GetCellNum(double* pv) //double* inverse primitive vector
{
    return abs( (int) ( pv[0]*(pv[4]*pv[8]-pv[5]*pv[7]) - pv[1]*(pv[3]*pv[8]-pv[5]*pv[6]) + pv[2]*(pv[3]*pv[7]-pv[4]*pv[6]) ));
}

class Info
{
public:

	double minVolume;
	double maxVolume;
	double threshold;
	double latticeMins[6];
	double latticeMaxes[6];
	const char* outputdir;

	bool forceMostGeneralWyckPos;
	double biasedrand;
	vector<int> biasedwycks;

	int spg;
	int spgnumber;

	double primitivector[9];
	double inversePrimitivector[9];
	int primitiveCellnum;
	char UselocalCellTrans;

	int maxAttempts;
	int maxattempts;
	int attemptstoGetCombs;

	int method;
	char fileformat;
	vector<Atoms> atomlist;
	vector<Structure> ans;
	vector<Structure> primitiveans;

	Info(double min = 0, double max = 0)
	{
		minVolume = min;
		maxVolume = max;
		for (int i = 0; i < 6; i++) latticeMins[i] = latticeMaxes[i] = 0;
		threshold = 0;
		forceMostGeneralWyckPos = true;
		biasedrand=1;
		maxAttempts = 1000;
		maxattempts = 500;
		attemptstoGetCombs = 0;

		method = 2;
		fileformat = 'v';
		UselocalCellTrans='y';
	}


	bool Check(double *m)
	{
		for (int i = 0; i < 6; i++) if (m[i] != 0) return false;
		return true;
	}

	void CompleteParm(Structure* s)
	{
		int temps=spg-1;
		for(int i=0;i<9;i++) primitivector[i]=primitive_vector_type[vector_type_choice[temps]-1][i];
		for(int i=0;i<9;i++) inversePrimitivector[i]=inverse_primitive_vector[vector_type_choice[temps]-1][i];
		primitiveCellnum=GetCellNum(inversePrimitivector);
		
		double maxr;
		if (minVolume == 0)
			minVolume = s->atomvolume * 1;
		if (maxVolume == 0)
			maxVolume = s->atomvolume * 3;
		if (threshold == 0)
			threshold = 1;
		if (Check(latticeMins))
		{
			if (s->maxr < 1.5)
			{
				for (int i = 0; i < 3; i++) { latticeMins[i] = 3.0; latticeMins[i + 3] = 60.0; }
				maxr = 3;
			}
			else
			{
				for (int i = 0; i < 3; i++) { latticeMins[i] = 2 * s->maxr; latticeMins[i + 3] = 60.0; }
				maxr = 2 * s->maxr;
			}
		}
		else maxr = 3;
		double maxlen = maxVolume / (maxr*maxr);
		if (Check(latticeMaxes))
		{
			if (maxlen > 3) for (int i = 0; i < 3; i++) { latticeMaxes[i] = maxlen; latticeMaxes[i + 3] = 120.0; }
			else for (int i = 0; i < 3; i++) { latticeMaxes[i] = 4.0; latticeMaxes[i + 3] = 120.0; }
		}

		//Here begins Get primitive cell transformed to conventional cell.
		maxVolume*=primitiveCellnum;
		minVolume*=primitiveCellnum;
		s->atomvolume*=primitiveCellnum;
		for(int i=0;i<3;i++)  {double tn=pow(primitiveCellnum,1.0/3);latticeMins[i]*=tn;latticeMaxes[i]*=tn;}
		for(int i=0;i<s->atoms.size();i++)
		{
			s->atoms[i].number*=primitiveCellnum;
			s->atoms[i].left*=primitiveCellnum;
			atomlist[i].number*=primitiveCellnum;
		}
			
		//Here ends Get primitive cell transformed to conventional cell.

		if (attemptstoGetCombs == 0)
		{
			if (method == 1) attemptstoGetCombs = sqrt(spgnumber) * 40;
			else if (method == 2) attemptstoGetCombs = 500;
		}
		return;
	};

	void CellTrans(vector<Structure>* ans, vector<Structure>* primitiveans)
	{
		for(int i=0;i<ans->size();i++)
		{
			Structure s;
			Structure* temps=&(*ans)[i]; double* lp=temps->latticeparm;
			s.volume=temps->volume/primitiveCellnum;
			s.spg = temps->spg;
			s.legal = temps->legal;
			s.UsedMostGeneral = temps->UsedMostGeneral;

			for(int j=0;j<3;j++)
			{
				s.latticeparm[j]=primitivector[j]*lp[0]+primitivector[j+3]*lp[1]+primitivector[j+6]*lp[2];
				s.latticeparm[j+3]=primitivector[j+3]*lp[4]+primitivector[j+6]*lp[5];
				s.latticeparm[j+6]=primitivector[j+6]*lp[8];
			}
			
			for(int j=0;j<temps->atoms.size();j++)
			{
				s.atoms.push_back(Atoms(temps->atoms[j].number/primitiveCellnum,temps->atoms[j].atom.name,temps->atoms[j].atom.radius,temps->atoms[j].atom.module));
				vector<position>* positions=&(s.atoms[j].positions);
				
				for (int k=0;k<temps->atoms[j].positions.size();k++)
				{
					position p=temps->atoms[j].positions[k];
					position tp;
					tp.x=p.x*inversePrimitivector[0]+p.y*inversePrimitivector[1]+p.z*inversePrimitivector[2];
					tp.y=p.x*inversePrimitivector[3]+p.y*inversePrimitivector[4]+p.z*inversePrimitivector[5];
					tp.z=p.x*inversePrimitivector[6]+p.y*inversePrimitivector[7]+p.z*inversePrimitivector[8];

					tp=Standford(&tp);

					if(positions->size()==0) positions->push_back(tp);
					else
						for(int a=0;a<positions->size();a++)
						{
							if(tp==(*positions)[a]) break;
							if(a==(positions->size()-1)) 
							{
								positions->push_back(tp);
								break;
							}
						}
					if(positions->size()==s.atoms[j].number) break;
					
				}
			}

			primitiveans->push_back(s);

		}
		return;
	};

	bool Generate(vector<Structure> &ans)
	{
		Structure structure;
		Initialize(structure, spg, atomlist, wyckpositions);
		CompleteParm(&structure);
		for (int i = 3; i < 6; i++)
		{
			latticeMins[i] = latticeMins[i] * M_PI / 180;
			latticeMaxes[i] = latticeMaxes[i] * M_PI / 180;
		}
		if (forceMostGeneralWyckPos == true)
		{
			bool l=false;
			for (int i = 0; i < atomlist.size(); i++)
				if (atomlist[i].number >= wyckpositions[wyckpositions.size() - 1].multiplicity)  {l = true; break;}
			if (l == false)
			{
				cout << "error: cannot generate a structure with most general wyckpos, turnning out the option may solve this problem" << endl;
				return false;
			}
		}
		vector<WyckGroup> wycks;
		GetWyckPosGrouped(wycks,0);
		structure.AddWyckGroup(&wycks);
		cout << "Initialize success: total time= " << 1.0*clock() / CLOCKS_PER_SEC << "s" << endl;

		vector<Structure> combinations;
		
		switch(method)
		{
		case 1:
		{
			int wsum=0;
			for(int i=0;i<wycks.size();i++)
			{
				biasedwycks.push_back(pow((*wycks[i].SimilarWyck)[0].multiplicity,biasedrand));
				wsum+=biasedwycks[i];
			} 

			for (int attempt = 0; attempt < attemptstoGetCombs; attempt++)
			{
				GetAllCombination(structure, wycks, combinations, forceMostGeneralWyckPos,&biasedwycks,wsum);
				if (combinations.size() >= sqrt(spgnumber) * 15) break;
			}
		}
		break;
		case 2: 
			GetAllCombinations(structure, wycks, combinations, forceMostGeneralWyckPos, attemptstoGetCombs);
		break;
		}
		
		//Here begins the logfile for combinations!
		/*for (int i = 0; i < combinations.size(); i++)
		{
			cout << "structure combination " << i << endl;
			for (int j = 0; j < combinations[i].atoms.size(); j++)
			{
				cout << combinations[i].atoms[j].atom.name << " : ";
				for (int k = 0; k < combinations[i].atoms[j].wyckGroups.size(); k++)
				{
					cout << combinations[i].atoms[j].wyckGroups[k].count << "(";

					for (int l = 0; l < (combinations[i].atoms[j].wyckGroups[k].SimilarWyck)->size(); l++)
						cout << (*combinations[i].atoms[j].wyckGroups[k].SimilarWyck)[l].multiplicity << (*combinations[i].atoms[j].wyckGroups[k].SimilarWyck)[l].label << ',';

					cout << ") , ";
				}
				cout << endl;
			}
		}*/
		//The logfile ends here.

		if (combinations.size() == 0)
		{
			ans.push_back(Structure(structure));
			cout << "error: Combination does not exit, total time= " << 1.0*clock() / CLOCKS_PER_SEC << "s" << endl;
			GetWycksDeleted(wycks);
			return false;
		}
		else cout << "GetAllCombination success: got " << combinations.size() << " combination(s); total time= " << 1.0*clock() / CLOCKS_PER_SEC << "s" << endl;
	

		int attemps = 0;
		int ans_size = 0;
		while (ans_size < spgnumber)
		{
			for (int j = 0; j < maxAttempts; j++)
			{
				structure = combinations[ rand() % combinations.size()];
				if (structure.ChooseLattice(latticeMins, latticeMaxes, minVolume, maxVolume)==false) continue;
				structure.ChooseWyck();
				//cout << "makecrystal start at " << 1.0*clock() / CLOCKS_PER_SEC << endl;
				structure.MakeCrystal(threshold, maxattempts);
				//cout << "makecrystal end at " << 1.0*clock() / CLOCKS_PER_SEC << endl;
				if (structure.legal == true)  break;
				if (j == maxAttempts - 1) 
				{ 
					attemps++; 
					ans_size++;
					cout << "error: failed MakeCrystal(), already made "<<ans.size()<<" crystal(s), total time= " << 1.0*clock() / CLOCKS_PER_SEC << "s" << endl;
				}
			}
			if (structure.legal == true)
			{
				Structure newstructure;
				Displace(newstructure, structure);
				ans.push_back(Structure(newstructure));
				ans_size++;
				//Here begins the logfile for crystal.
				/*ofstream out("log_structurecombs.txt",ios::app);
				out << "spg= "<<spg<<", structure= " << ans.size()  << '\n';
				for (int i = 0; i < structure.atoms.size(); i++)
				{
					out << structure.atoms[i].atom.name << " : ";
					for (int j = 0; j < structure.atoms[i].chosenWycks.size(); j++)
						out << wyckpositions[structure.atoms[i].chosenWycks[j]].multiplicity << wyckpositions[structure.atoms[i].chosenWycks[j]].label<<",";
					out << '\n';
				}
				out.close();*/
				//And here it ends.
			}
			if (attemps > 5) 
			{
				GetWycksDeleted(wycks); 
				if (ans.size() > 0)
				{
					cout<<"Notice: exit for too many MakeCrystal() failures; " << ans.size() << " crystal(s) were generated in total." << endl;
					return true;
				}
				else return false;
			}
		}
		GetWycksDeleted(wycks);
		return true;
	};

	void AppendAtoms(int num,const char* name,double radius,bool m)
	{
		atomlist.push_back(Atoms(num, name, radius,m));
		return;
	}

	bool PreGenerate(void)
	{
		srand((unsigned)time(NULL));
		bool legel = Generate(ans);
		switch(UselocalCellTrans)
		{
		case 'y':
			CellTrans(&ans,&primitiveans);
			break;
		case 'n':
		{
			for(int i=0;i<ans.size();i++)
				primitiveans.push_back(ans[i]);
		}
			break;
		}
		
		if (legel)
		{
			cout << "Generate success: total time= " << 1.0*clock() / CLOCKS_PER_SEC << "s" << endl;
			
			/*string output(outputdir);
			for (int i = 0; i < primitiveans[0].atoms.size(); i++)
			{
				output.append(primitiveans[0].atoms[i].atom.name);
				output.append(to_string(primitiveans[0].atoms[i].number));
			}
			output.append("_"); output.append(to_string(spg)); output.append("-");
			for (int i = 0; i < primitiveans.size(); i++)
			{
				string filename(output);
				filename.append(to_string(i + 1));
				if(fileformat=='g') filename.append(".gin");
				primitiveans[i].WritePoscar(&filename,fileformat);
				/*filename.append("-Cell.py");
				ans[i].WritePoscar(&filename,fileformat);
			}*/
		}
		else cout << "error: Generate error" << endl;
		return legel;
	}

	p::list GetLattice(int n)
	{
		Py_Initialize();
		np::initialize();
		p::list l;
		if (n >= primitiveans.size()) { cout << "Please input a smaller number than " << primitiveans.size() << endl; return l; }
		for (int i = 0; i < 9; i++) l.append(primitiveans[n].latticeparm[i]);
		return l;
	}
	void GetAtom(int n)
	{
		if (n >= primitiveans.size()) { cout << "Please input a smaller number than " << primitiveans.size() << endl; }
		cout<<"There's "<<primitiveans[n].atoms.size()+1<<" type(s) of atoms in this structure."<<endl;
		for (int i = 0; i < primitiveans[n].atoms.size(); i++)
		{
			cout<<primitiveans[n].atoms[i].atom.name<<'\t'<<primitiveans[n].atoms[i].positions.size()+1<<endl;
		}

		return ;
	}
	p::list GetPosition(int n)
	{
		Py_Initialize();
		np::initialize();
		p::list l;
		if (n >= primitiveans.size()) { cout << "Please input a smaller number than " << primitiveans.size() << endl; return l; }
		for (int i = 0; i < primitiveans[n].atoms.size(); i++)
		{
			for (int j = 0; j < primitiveans[n].atoms[i].positions.size(); j++)
			{
				l.append(primitiveans[n].atoms[i].positions[j].x);
				l.append(primitiveans[n].atoms[i].positions[j].y);
				l.append(primitiveans[n].atoms[i].positions[j].z);
			}
		}

		return l;
	}

	void SetLatticeMins(double a, double b, double c, double d, double e, double f)
	{
		latticeMins[0] = a; latticeMins[1] = b; latticeMins[2] = c;
		latticeMins[3] = d; latticeMins[4] = e; latticeMins[5] = f;
		return;
	}
	void SetLatticeMaxes(double a, double b, double c, double d, double e, double f)
	{
		latticeMaxes[0] = a; latticeMaxes[1] = b; latticeMaxes[2] = c;
		latticeMaxes[3] = d; latticeMaxes[4] = e; latticeMaxes[5] = f;
		return;
	}
};

/*int main()
{
	for (int i =1; i <= 230; i++)
	{
		Info info;
		info.spg=i;
		cout << "spg=" <<info.spg<< ": start at "<<1.0*clock()/CLOCKS_PER_SEC<<"s"<<endl;

		info.AppendAtoms(12, "Ti", 1.6,false);
		info.AppendAtoms(48, "O", 0.66,false);
		//info.AppendAtoms(15, "Mg", 1.41,false);
		//info.AppendAtoms(15, "Si", 1.11,false);

		info.minVolume = 590;
		info.maxVolume = 610;
		info.maxAttempts = 1000;
		info.spgnumber = 10;
		info.threshold = 0.5;
		info.forceMostGeneralWyckPos = true;
		info.method =2;
		info.outputdir = "outputtest/";
		info.fileformat='t';
		//info.biasedrand=3;
		double mins[6] = { 3,3,3,60,60,60 };
		for (int j = 0; j < 6; j++) info.latticeMins[j] = mins[j];
		double maxs[6] = { 10,10,10,120,120,120 };
		for (int j = 0; j < 6; j++) info.latticeMaxes[j] = maxs[j];
		info.PreGenerate();
	}
	return 0;
}*/




BOOST_PYTHON_MODULE(GenerateNew)
{
	using namespace boost::python;
	class_<Info>("Info")
		.def_readwrite("minVolume", &Info::minVolume)
		.def_readwrite("maxVolume", &Info::maxVolume)
		.def_readwrite("threshold",&Info::threshold)
        .def_readwrite("outputdir",&Info::outputdir)

		.def_readwrite("spg", &Info::spg)		
		.def_readwrite("spgnumber", &Info::spgnumber)
        .def_readwrite("maxAttempts", &Info::maxAttempts)

		.def_readwrite("forceMostGeneralWyckPos",&Info::forceMostGeneralWyckPos)
        .def_readwrite("biasedrand",&Info::biasedrand)

		.def_readwrite("method",&Info::method)
        .def_readwrite("fileformat",&Info::fileformat)
		.def_readwrite("UselocalCellTrans",&Info::UselocalCellTrans)

		.def("AppendAtoms", &Info::AppendAtoms)
		.def("PreGenerate", &Info::PreGenerate)
		.def("GetLattice", &Info::GetLattice)
		.def("GetAtom", &Info::GetAtom)
		.def("GetPosition", &Info::GetPosition)
		.def("SetLatticeMins", &Info::SetLatticeMins)
		.def("SetLatticeMaxes", &Info::SetLatticeMaxes)
		;
}