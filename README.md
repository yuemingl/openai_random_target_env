# A user defined environment for OpenAI Gym

This is a user defined environment. Given a start point (0,0) and a random target point (x1, y1),
the agent takes actions 0,1,2,3 (forward, backword, turn left, turn right) and step length 0.1 to reach the given target point

### Install virtual env and tensorflow
```sh
source ~/venv14/bin/activate
pip install --upgrade tensorflow==1.14
```

### Train
```sh
python runRandomTarget.py
```

### Test
```sh
python runRandomTarget.py --mode test --weight model.save
```

### Test result
```sh
step= 0  current_pos=( 0.0 , 0.1 ) target=( 9.0 , 3.0 )
step= 1  current_pos=( 0.1 , 0.1 ) target=( 9.0 , 3.0 )
step= 2  current_pos=( 0.2 , 0.1 ) target=( 9.0 , 3.0 )
...
step= 198  current_pos=( 9.0 , 2.9 ) target=( 9.0 , 3.0 )
step= 199  current_pos=( 9.0 , 3.0 ) target=( 9.0 , 3.0 )
```
