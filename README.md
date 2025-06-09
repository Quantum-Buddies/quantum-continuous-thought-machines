# Quantum Continuous-Thought Machine

This repository contains a PyTorch-based implementation of a Hybrid Quantum-Classical Continuous-Thought Machine (CTM), inspired by the work from [Sakana AI](https://github.com/SakanaAI/continuous-thought-machines). This model integrates quantum circuits into the core recurrent memory mechanism of a CTM, exploring the potential of quantum machine learning for complex sequential tasks.

### CTM Thought Process Visualization

Here we see the model "thinking" about a particularly challenging problem: identifying a rabbit on the University of Leeds campus. It's a tough job, but someone's gotta do it!

![CTM Visualization](./figures/ctm_visualization.gif)

## Architecture Overview

The `HybridCTM` is a modular framework designed to solve a variety of tasks by leveraging a recurrent "thought" process. The core of the model consists of a set of quantum memory slots that are updated and synchronized over a series of iterations. An attention mechanism, guided by a quantum synchronization vector, allows the model to focus on relevant parts of the input. The architecture is designed to be flexible, with different backbones and output heads for different tasks.

![Architecture Diagram](./architecture_diagram.md)

### Key Components

*   **Backbone**: A standard neural network (e.g., ResNet-18 for images or a Linear layer for vectors) that processes the initial input into a feature vector.
*   **Quantum Memory**: A set of `QuantumNeuronLevelModels` (QNLMs) that act as the recurrent memory of the CTM. Each QNLM is a quantum circuit implemented using either **PennyLane** or **Qiskit**.
*   **Quantum Synchronization Layer**: A novel component that measures the correlation between pairs of quantum memory slots. This produces a "synchronization vector" that guides the attention mechanism.
*   **Attention Mechanism**: A standard `nn.MultiheadAttention` layer that uses the synchronization vector to query the input features, allowing the model to dynamically focus its attention during its thought process.
*   **Task-Specific Heads**: The final output is produced by different heads depending on the task. For classification and maze-solving, a single output projector is used. For reinforcement learning, separate actor and critic heads are used.

## Implemented Tasks

This project currently includes implementations for three tasks:

1.  **Image Classification**: Training the CTM to classify MNIST digits.
2.  **Reinforcement Learning**: Using the CTM as the policy and value network for a PPO agent in the CartPole environment.
3.  **Maze Solving**: Training the CTM to navigate and solve mazes.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Quantum-Buddies/quantum-continuous-thought-machines.git
    cd quantum-continuous-thought-machines
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

You can train the CTM on any of the implemented tasks using the scripts in the `tasks` directory.

### 1. Image Classification (MNIST)

This task trains the CTM to classify handwritten digits from the MNIST dataset.

```bash
python tasks/image_classification/training_loop.py --epochs 10 --batch_size 64 --backend pennylane
```

**Key Arguments:**
*   `--epochs`: Number of training epochs.
*   `--batch_size`: Training batch size.
*   `--lr`: Learning rate.
*   `--backend`: The quantum backend to use (`pennylane` or `qiskit`).

### 2. Reinforcement Learning (CartPole)

This task uses the CTM as the policy for a PPO agent to solve the CartPole-v1 environment.

```bash
python tasks/rl/train_rl.py --total-timesteps 50000 --backend pennylane
```

**Note on Backends**: The PennyLane backend is significantly faster for this task due to its more efficient gradient calculation methods. The Qiskit backend is supported but will be very slow.

**Key Arguments:**
*   `--total-timesteps`: The total number of timesteps to train for.
*   `--num-envs`: The number of parallel environments to use.
*   `--backend`: The quantum backend to use (`pennylane` or `qiskit`).

### 3. Maze Solving

This task trains the CTM to find the solution path in a maze.

**First, you need to download the maze dataset from the original CTM repository and place it in the `data/mazes/medium` directory.**

```bash
python tasks/mazes/train_mazes.py --epochs 10 --batch_size 4 --backend pennylane
```

**Key Arguments:**
*   `--data_root`: The root directory of the maze dataset.
*   `--epochs`: Number of training epochs.
*   `--batch_size`: Training batch size.
*   `--backend`: The quantum backend to use (`pennylane` or `qiskit`).

## Project Status

This is a functional, end-to-end implementation of a Quantum Continuous Thought Machine. The model successfully trains and demonstrates the architectural principles.

## Acknowledgements

This work is based on the paper "[Continuous-Time Thought Processes](https://arxiv.org/abs/2405.16829)" by Sakana AI. 