# Shared Autonomy Experiment of Robot Trajectron V2

This repository contains code for the RT-V2 workspace.

[[paper Link](https://arxiv.org/abs/2509.19954)]
[[project page](https://mousecpn.github.io/RTV2_page/)]
[[model training](https://github.com/mousecpn/Robot-TrajectronV2)]

## 🛠️ Prerequisites

- Install the environment for [Robot Trajectron V2](https://github.com/mousecpn/Robot-TrajectronV2).
- Install `zmq`, `pybullet`

## 🚀 Navigation Experiment
```bash
python simple_planning_env.py
```

You can set `viz=True` in the code to save the navigation images as follows.

<img width="2558" height="416" alt="image" src="https://github.com/user-attachments/assets/ca662ffd-df0b-4bec-b52b-ac9e77f6f85d" />


## 🤖 Shared Control Experiment Setup

1. **Activate RT-V2:**  
    ```bash
    python TrajectronNode.py
    ```
2. **Launch environment:**  
    ```bash
    python sim_env.py
    ```

The blue box is the target, green boxes are distractors, and the red boxes are the obstacles.
You can use the ⬆️,⬇️,⬅️,➡️ on the keyboard to control the ball. The trial will fail when (i) the ball collides with the obstacles; (ii) your control iterations are out of 200.

<img width="1044" height="823" alt="Screenshot from 2026-01-22 10-56-29" src="https://github.com/user-attachments/assets/757d25e8-7002-4a6f-8e44-dcc567eb1fcf" />


## 📄 License

See [LICENSE](LICENSE) for details.
