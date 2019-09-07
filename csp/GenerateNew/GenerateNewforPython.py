#-----preset-----
#spg=1
#spgnumber=5
#maxAttempts=1000
#minVolume=atomvolum*1; maxVolume=atomvolume*3
#latticeMins=[3.0,3.0,3.0,60.0,60.0,60.0] or [2*s.maxr,2*s.maxr,2*s.maxr,60.0,60.0,60.0]
#latticeMaxes=[4.0,4.0,4.0,120.0,120.0,120.0] or [maxlen,maxlen,maxlen,120.0,120.0,120.0]
#threshold=1

#-----function-----
#-----Firstly, generate class Info()-----
import GenerateNew
for s in range(143,144):
    print('spg='+str(s))
    temp=GenerateNew.Info()
#-----How to set parameters-----
#Set directly:
    temp.minVolume=490
    temp.maxVolume=510
    temp.threshold=0.57
    temp.spg=s
    temp.maxAttempts=1000
    temp.spgnumber=4
    
    temp.method=2
    temp.forceMostGeneralWyckPos=True
#Set through functions:
#--temp.AppendAtoms(int number_of_atoms, const char* atom_name, double radius,bool legal)
    temp.AppendAtoms(48,"Ti",1.6,False)
    temp.AppendAtoms(24,"O",0.66,False)
    temp.SetLatticeMins(3,3,3,60,60,60)
    temp.SetLatticeMaxes(10,10,10,120,120,120)
    #temp.outputdir="Ti12O48/"
    temp.UselocalCellTrans='y'
    #temp.fileformat='v'
#-----function!-----
    numofStructuresGenerated=temp.PreGenerate()
    print("Generated "+str(numofStructuresGenerated)+" strucctures\n")
#-----Get Results-----
#There will be "spgnumber" structures generated
#Use GetLattice(i),GetAtom(i) andGetPosition(i) to get the i_th structure
#GetAtom(i) prints name,number,name,number,...
#GetPosition(i) returns name1.x, name1.y, name1.z, name1.x, name1.y, name1.z, ...until no name1 left
#then for name2, name2.x, name2.y, name2.z, name2.x, name2.y, name2.z...
for i in range(numofStructuresGenerated):
    print("\n"+str(i)+":")
    a=temp.GetLattice(i)
    print (a) 
    print(len(a))
   
    #temp.GetAtom(i)
    #print('\n')
    b=temp.GetPosition(i)
    print(len(b))