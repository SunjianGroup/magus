# Magus



## 安装

### 依赖

- Python 3
- Numpy
- Scipy
- scikit-learn
- ASE
- spglib
- pandas
- networkx


在集群上加载`anaconda/3` 模块，所有依赖库都已安装




### 安装过程

- 加载`Anaconda`:

  ```shell
  module add anaconda/3
  ```

- 在`~/.bashrc`中设置路径：

  ```shell
  export PYTHONPATH=$PYTHONPATH:/your/path/magus
  export CSP_TOOLS=/your/path/magus/tools
  export PATH=$PATH:$CSP_TOOLS
  ```

- 设置`ASE`的VASP calculator:

  建一个`run_vasp.py`:

  ```python
  import subprocess
  exitcode = subprocess.call("mpiexec.hydra /your/path/to/vasp", shell=True)
  ```

  建立`mypps`目录存放vasp赝势，可以用软连接：

  ```
  mypps/
  ├── potpaw
  ├── potpaw_GGA
  └── potpaw_PBE
  ```
  ```shell
  ln -s /your/path/PBE-5.4 mypps/potpaw_PBE
  ```

  三个子目录分别对应LDA, PW91, PBE

  也可以加入其他赝势库。

  设置环境变量：

  ```shell
  export VASP_SCRIPT=/your/path/run_vasp.py
  export VASP_PP_PATH=/your/path/mypps
  ```

  更多信息见 https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#module-ase.calculators.vasp

**注意：`run_vasp.py`和`mypps`最好不要放在`magus`目录下**

- 编译库文件

  以前的云盘压缩包提供了库文件`fmodules.so`和`GenerateNew.so`，可以继续使用。现在的git库中只有源代码，如果没有库文件或者由于某些原因出现问题（例如Python版本变化），需要重新编译：
  - `fmodules.so`
      在`csp/`下运行`f2py -c -m fmodules fmodules.f90`
  - `GenerateNew.so`
      源文件在在`csp/GenerateNew`中。编译时需要用python库的头文件。如果使用集群上的`anaconda/3` 模块，编译命令为：
      ```shell
      g++ -std=c++11 -I/fs00/software/anaconda/3/include -I/fs00/software/anaconda/3/include/python3.6m -L/fs00/software/anaconda/3/lib -lboost_python -lboost_numpy -lpython3.6m GenerateNew.cpp -o GenerateNew.so -shared -fPIC
      ```
      编译生成的`GenerateNew.so`需要放在`csp/`目录下。
      



## 输入文件

- `inputFold/`: 结构优化输入文件的目录，使用VASP时，把INCAR文件保存为`inputFold/INCAR_*`
- `fpFold/fpsetup.yaml`: 计算结构指纹的设置，一般不用改动
- `input.yaml`

### input.yaml的设置

- `input.yaml`例子：

``` yaml
calcType: fix
calculator: vasp
setAlgo: bayes
xc: PBE
popSize: 20
numGen: 20
minAt: 6
maxAt: 12
symbols: ['Ti', 'O']
ppLabel: ['_sv', '']
formula: [1, 2]
dRatio: 0.7
randFrac: 0.2
saveGood: 5
pressure: 0
addSym: False
calcNum: 5
numParallel: 2
numCore: 12
queueName: e52692v2ib!
waitTime: 200

### Bayesian
kappa: 2
kappaLoop: 2
scale: 0.0005
parent_factor: 0.1

### BBO
grids: [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
migrateFrac: 0.4
mutateFrac: 0.4
```

#### 参数介绍

- calcType: 计算类型

  可用值:  `fix`（定组分）,`var`（变组分）

- calculator: 结构优化程序
  可用值: `vasp`, `gulp`

- setAlgo: 结构搜索算法
  可用值: `bayes`(Bayesian Optimization), `bbo`(Biogeography-Based Optimization)

- xc: 交换关联类型

  可用值: `PBE`, `LDA`,`PW-91`

- spacegroup: 随机结构的空间群

  例：[1,2,20-30]

- initSize: 初代种群数量
- popSize: 种群数量
- numGen: 迭代次数
- minAt: 最小原子数
- maxAt: 最大原子数
- symbols: 元素类型

  例：['Ti', 'O'], 外层是方括号，每个元素用引号括起来

- ppLabel: VASP赝势的后缀

  例：['_sv', ''], 与symbols顺序一致，若无后缀则填入''

- formula: 元素比例

  例： [1, 2]

- fullEles: 若值为`True`,则产生的结构含有'symbols'中所有元素，只在变组分搜索时生效
- eleSize: 变组分搜索时，初代每种单质随机产生的结构数
- volRatio: 随机产生结构时的体积参数
- randFrac: 随机结构比例
- dRatio: 判断原子距离是否过近的标准
- saveGood: 保留结构数
- pressure: 压强(GPa)
- addSym: 产生结构之前是否为父代加入对称性
- molDetector: 结构演化时判断分子片段的方法
  可用值：0(不判断分子局域结构，默认值)  1(自动判断分子局域结构，建议使用)  2(使用Girvan-Newman算法划分局域结构)
- exeCmd: 运行结构优化程序的命令（只有`calculator`为`gulp`时才需要）
- calcNum: 结构优化次数
- numParallel: 并行优化结构的数目
- numCore: 结构优化使用的核数
- queueName: 结构优化任务的队列
- jobPrefix: 并行模式下任务脚本的前缀
- waitTime: 检查结构优化任务的时间间隔

##### Bayesian Optimization 参数

- parent_factor: 父代能量的系数，该参数越大，越接近进化算法
- kappa: 2
- kappaLoop: 2
- scale: 0.0005

  ​

##### BBO 参数

- grids: 切割晶胞的网格
  例：[[2, 1, 1], [1, 2, 1], [1, 1, 2]], 两层方括号

- migrateFrac: 迁移操作产生结构的数目
- mutateFrac: 变异算子产生结构的数目

#### 部分参数默认值

- spacegroup: list(range(1, 231))
- eleSize: 1
- fullEles: False
- volRatio: 1.5
- dRatio: 0.7
- exeCmd: ""
- initSize: parameters['popSize']
- jobPrefix:""
- permNum: 4
- latDisps: list(range(1,5))
- ripRho: [0.5, 1, 1.5, 2]
- molDetector: 0
- rotNum: 5

更多默认参数见`readparm.py`

## 输出文件
输出文件保存在`results`目录下，分为四种：
- `gen*.traj`: 每代所有结构
- `pareto*.traj`: 每代最优结构
- `keep*.traj`: 每代保留到下一代的结构
- `good.traj`: 当前最好的`popSize`个结构

提取结构信息需要`summary.py`, 运行方式：`summary.py good.traj`



## 计算流程

以 **TiO**$_2$结构搜索为例：

- 运行`csp-prepare`, 产生`BuildStruct`, `fpFold`, `inputFold`三个目录以及`summary.py`

- 准备输入文件`input.yaml`（如上所示）和`INCAR_*`(1-5), 把`INCAR_*`放入`inputFold`/

- 确认在`~/.bashrc`中已经加载了anaconda模块

- 提交任务脚本：

  ```shell
  #BSUB -q e52692v2ib!
  #BSUB -n 1
  #BSUB -J test-tio2

  python -m csp.parallel 
  ```
  
- 若不想在`~/.bashrc`中加载anaconda模块，则需要在任务脚本中加入anaconda，如`module add anaconda/3`，并将`jobPrefix`也设为`module add anaconda/3`

- 运行过程中的输出信息保存在`log.txt`中，所有结构信息都保存在`results`目录下



## 注意事项
- 在.bashrc中配置路径时，不要复制pdf文档中的字符，可能会有无法识别的空白字符，最好直接复制markdown文件的内容
- 用vasp优化时，需要在INCAR文件中设置KSPACING，与USPEX不同