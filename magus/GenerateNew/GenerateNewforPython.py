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
for s in range(128,129):
    print('spg='+str(s))
    temp=GenerateNew.Info()
#-----How to set parameters-----
#Set directly:
    temp.minVolume=290
    temp.maxVolume=310
    temp.threshold=0.5
    temp.spg=s
    temp.maxAttempts=1000
    temp.spgnumber=1

    temp.method=2
    temp.forceMostGeneralWyckPos=False
#Set through functions:
#--temp.AppendAtoms(int number_of_atoms, const char* atom_name, double radius,bool legal)
    temp.AppendAtoms(8,"0",3.1,False)
    # temp.AppendAtoms(20,"1",0.66,False)
    temp.SetLatticeMins(3,3,3,60,60,60)
    temp.SetLatticeMaxes(10,10,10,120,120,120)
    #temp.outputdir="Ti12O48/"
    # temp.UselocalCellTrans='y'
    temp.GetConventional=True
#if you want to go back to the old version, set it to true
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
#GetWyckPos(i) returns name1.name, name1.x, name1.y, name1.z...
for i in range(numofStructuresGenerated):
    print("\n"+str(i)+":")
    a=temp.GetLattice(i)
    print (a)
    #temp.GetAtom(i)
    #print('\n')
    b=temp.GetPosition(i)

    print(len(b)/3)
    for j in range(len(b)):
        print(b[j],end=' ')
        if(j%3==2):
            print(' ')
    c=temp.GetWyckPos(i)
    print("Next:")
    for j in range(len(c)):
        print(c[j],end=' ')
        if(j%4==3):
            print(' ')

