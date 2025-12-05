# VAMAPPO: Value-Adaptive Multi-Agent PPO for GenAIaaS Orchestration

This repository contains the reference implementation of the VAMAPPO (Value-Adaptive Multi-Agent Proximal Policy Optimization) framework, designed for collaborative Edge AI inference and GenAI-as-a-Service (GenAIaaS) orchestration.

The framework addresses the challenges of stochastic latency spikes and complex microservice dependencies in distributed edge environments by introducing a value-adaptive mechanism and a Coordinated Advantage Function (CAF).

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

## Note on Implementation

This repository provides a prototype implementation focused on demonstrating the software architecture and the interaction flow of the VAMAPPO framework.

Please note the following regarding the comparison with the paper:
- Environment Simplified: The network topology and latency models in this codebase use simplified geographical distance metrics for demonstration purposes, rather than the graph-based transmission models described in the paper.
- Logic Abstraction: Specific mathematical formulations (e.g., the exact coefficients for the Coordinated Advantage Function and the specific Sigmoid utility parameters) have been simplified to generic linear forms to facilitate ease of understanding and modification.

Researchers intending to reproduce the exact numerical results reported in the paper should refer to the mathematical models detailed in the System Model and Problem Formulation sections of the manuscript.

## License

This project is licensed under the MIT License.

