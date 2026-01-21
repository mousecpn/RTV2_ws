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

## 🤖 Shared Control Experiment Setup

1. **Activate RT-V2:**  
    ```bash
    python TrajectronNode.py
    ```
2. **Launch environment:**  
    ```bash
    python sim_env.py
    ```

## 📄 License

See [LICENSE](LICENSE) for details.