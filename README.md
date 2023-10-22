## Distributed Multi-vehicle Task Assignment and Motion Planning in Dense Environments
Python Implementation of multi-vehicle task assignment and motion planning (MVTAMP), including task assignment algorithms: consensus-based bundle algorithm (CBBA), lazy sample-based task allocation (LSTA), decentralized sample-based task allocation (DSTA), and our lazy-based review consensus algorithm (LRCA); motion planning algorithms: optimal reciprocal collision avoidance (ORCA), D*, and our guidance point strategy (GOS). Finally, a novel hierarchical method, LRGO, for solving the MVTAMP applied in non-holonomic vehicles  is presented in this repository.

-----

Description
-----

We present an novel method for solving the multi-vehicle task assignment and motion planning (MVTAMP) problem for non-holonomic vehicles, in which a fleet of non-holonomic vehicles is assigned to visit a series of targets to clear them efficiently, and each vehicle needs to travel to a specific ending area once it finished all missions in the shortest possible time.

About
-----

**Paper**:  Distributed Multi-vehicle Task Assignment and Motion Planning in Dense Environments, Gang Xu, Xiao Kang, Helei Yang, Yuchen Wu, Junjie Cao,  Weiwei Liu, and Yong Liu


-----

Requirement
-----

```python
pip install numpy
pip install pandas
pip install random
pip install matplotlib
```

-----

Applications
-----

```python
cd run_example
python run_exam_gorca_s1.py
In the file run_example, you can select the pyhton scripts to test different scenarios. 
```

#### The first simulation: 50 vehicles visit 203 targets for clearing them and move to the start area.

<video id="video" controls=""src="draw/videos/p1.mp4" preload="none">

#### The results of evaluation.

<p align="center">
    <img src="draw/figs/evaluation.png" width="800" height="400" />
</p>


#### The first real-world experiment: 5 vehicle visit 11 targets.

<video id="video" controls=""src="draw/videos/p2.mp4" preload="none">




#### The first real-world experiment: 5 vehicle visit 11 targets and move to the start area.

<video id="video" controls=""src="draw/videos/p3.mp4" preload="none">



----

Discussion
----

Papers: [CBBA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5072249); [DSTA](https://link.springer.com/article/10.1007/s11721-022-00213-0); [LSTA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8798293)
