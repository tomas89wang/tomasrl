# tomasrl

gym 仿真命令跟 vi 方向键一样

k 向上  
j 向下  
h 向左  
l 向右  
^C 或者 ^D 退出  

格子世界中  
@ 代表初始状态  
$ 代表终止状态  
\# 代表当前状态  

风格世界  
$ python -m models.cliff_grid_world  

悬崖世界  
$ python -m models.wind_grid_world  

21点  
$ python -m models.black_jack  

测试 sarsa 算法  
$ python test_sarsa.py WindGridWorld-v0  
$ python test_sarsa.py CliffGridWorld-v0  

测试 sarsa-lambda 算法  
$ python test_lambda.py WindGridWorld-v0  
$ python test_lambda.py CliffGridWorld-v0  

测试 q-learn 算法  
$ python test_q_learn.py WindGridWorld-v0  
$ python test_q_learn.py CliffGridWorld-v0  
