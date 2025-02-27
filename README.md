Embedded Control on Equivariant Manifold Flows
===

This is the implementation of the paper [Koopman Embedded Equivariant Control](https://arxiv.org/abs/2312.01544).

An efficient way to control systems with unknown nonlinear dynamics is to find an appropriate embedding or representation for simplified approximation (e.g. linearization), which facilitates system identification and control synthesis. Nevertheless, there has been a lack of embedding methods that can guarantee (i) embedding the dynamical system comprehensively, including the vector fields (ODE form) of the dynamics, and (ii) preserving the consistency of control effect between the original and latent space. To address these challenges, we propose Koopman Embedded Equivariant Control (KEEC) to learn an embedding of the states and vector fields such that a Koopman operator is approximated as the latent dynamics. Due to the Koopman operator's linearity, learning the latent vector fields of the dynamics becomes simply solving linear equations. Thus in KEEC, the analytical form of the greedy control policy, which is dependent on the learned differential information of the dynamics and value function, is also simplified. Meanwhile, KEEC preserves the effectiveness of the control policy in the latent space by preserving the metric in two spaces. Our algorithm achieves superior performances in the experiments conducted on various control domains, including the image-based Pendulum, Lorenz-63 and the wave equation.
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
