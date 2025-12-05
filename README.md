# A Value-Adaptive Multi-Agent PPO for GenAIaaS Orchestration

This repository contains the reference implementation of the Value-Adaptive Multi-Agent Proximal Policy Optimization framework, designed for collaborative Edge AI inference and GenAI-as-a-Service (GenAIaaS) orchestration.

The framework addresses the challenges of stochastic latency spikes and complex microservice dependencies in distributed edge environments by introducing a value-adaptive mechanism and a coordinated advantage function.

## System Architecture

The system models the GenAI orchestration problem as a Networked Markov Decision Process (MDP). The architecture consists of:

1.  Distributed Environment: Simulates a heterogeneous edge network with varying computational capabilities and link constraints.
2.  Value-Adaptive Agents: Implements PPO agents that dynamically adjust learning rates and clipping ranges based on uncertainty quantification.
3.  Multi-Agent Coordination: Facilitates cooperation among distributed agents to optimize the end-to-end service utility.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Gymnasium
- NumPy, Pandas, Matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/VAMAPPO.git
cd VAMAPPO

# Install dependencies
pip install -r requirements.txt
```

### Running the Training

To start the multi-agent training process:

```bash
python src/distributed_main.py --mode train --n_agents 5
```

To run the inference evaluation:

```bash
python src/distributed_main.py --mode inference --model_path results/
```

## License

This project is licensed under the MIT License.



