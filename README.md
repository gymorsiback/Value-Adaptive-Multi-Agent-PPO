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
## Dataset Origin

This dataset is derived from **OpenCellID-grounded topologies**, which are constructed using cellular infrastructure data from **OpenCellID**: [https://opencellid.org/#zoom=16&lat=37.77889&lon=-122.41942](https://opencellid.org/#zoom=16&lat=37.77889&lon=-122.41942)

## Dataset Description

### 1. Server Topologies

The dataset includes four primary server-topology configurations (**S1–S4**), each representing a distinct deployment scale and scenario:

#### S1 — Switzerland Region (Small-Scale, High-Density)

* **Number of nodes**: 25 distributed server nodes
* **Compute capacity**: 16–54 units (mean: 34.6 units)
* **Storage capacity**: 16–34 units (mean: 25.3 units)
* **Network connectivity**: sparse graph with an average degree of 3.2
* **Deployment type**: regional datacenter-style deployment
* **Intended use**: evaluation under small-scale, high-density deployments

#### S2 — United Kingdom Region (Medium-Scale, Geographically Distributed)

* **Number of nodes**: 51 distributed server nodes
* **Compute capacity**: 16–53 units (mean: 35.8 units)
* **Storage capacity**: 15–38 units (mean: 26.2 units)
* **Deployment type**: nation-scale infrastructure
* **Intended use**: evaluation under medium-scale geographically distributed deployments

#### S3 — Germany Region (Large-Scale, Heterogeneous)

* **Number of nodes**: 101 distributed server nodes
* **Compute capacity**: 19–55 units (mean: 36.7 units)
* **Storage capacity**: 12–39 units (mean: 26.0 units)
* **Deployment type**: large-scale heterogeneous-resource environment
* **Intended use**: validation of orchestration complexity under large-scale deployments

#### S4 — Milan Metropolitan Network (Urban Edge Computing)

* **Number of nodes**: 31 densely deployed servers
* **Deployment type**: urban edge-computing scenario
* **Intended use**: evaluation of edge intelligence in smart-city settings

### 2. Model Deployment

Each topology is accompanied by a deployment of generative models distributed across nodes to emulate heterogeneous real-world placements:

* **Total models per topology**: 227 generative models
* **Text-to-Text models**: 191
* **Text-to-Image models**: 30
* **Image-to-Video models**: 6

**Note:** To reflect practical heterogeneity, the model placement strategy differs across topologies.

### 3. Task Dataset and Train/Test Splits

Each topology includes user-task traces for training and evaluation:

* **Training set**: 1,000 user tasks per topology
* **Test set**: 200 independent tasks per topology
* **Special setting**: **S4** is additionally designated as a test-only topology for **cross-domain generalization** evaluation.

---
## Contact & Support

For questions, bug reports, or collaboration inquiries:

- **Email**: [gymorsiback@tju.edu.cn]

---

## License

This project is licensed under the MIT License.




