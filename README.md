Embedded Control on Equivariant Manifold Flows
===

This is the implementation of the paper [Embedded Control on Equivariant Manifold Flows](https://arxiv.org/abs/2312.01544)

## Control Demos
### Pendulum 
![Pendulum control](figures/pendulum.gif)
### Lorenz-63
![Lorenz control](figures/lorentz_attractor_keec.gif)
### Wave equation
#### controlled wave equation
![KEEC controlled wave equation](figures/keec_controlled_wave.gif)
#### uncontrolled wave equation
![Uncontrolled wave equation](figures/uncontrolled_wave.gif)

## Install & Dependence
First, clone the repository:
```
git clone https://github.com/yyimingucl/Koopman-Embed-Equivariant-Control.git
```
Then install the dependencies as listed in ```environment.yml``` and activate the environment: 
```
conda env create -f environment.yml
conda activate koopman_policy
```
## Use
- for train
  ```
  python training_scripts/task_name/main.py
  ```
- for test
  ```
  python training_scripts/task_name/evaluate.py
  ```
The settings can be changed in the ```training_scripts/task_name/config.yaml```. We provide an example notebook with trained model in ```wave_example.ipynb```.
## Pretrained model
| Model | Task |
| ---     | ---   |
| trained_weights | wave |



## Directory Hierarchy
```
|—— convex_solver.py
|—— dataset.py
|—— environment.yml
|—— figures
|—— model.py
|—— requirements.txt
|—— roll_out.py
|—— train.py
|—— trained_weights
|—— training_scripts
|    |—— pendulum-gym
|        |—— config.yaml
|        |—— evaluate.py
|        |—— main.py
|    |—— wave
|        |—— config.yaml
|        |—— evaluate.py
|        |—— main.py
|—— utils.py
|—— wave_example.ipynb
|—— wrappers.py
```
