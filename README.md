Automated High-Fidelity Restaurant Marketing and Digital System

Enterprise Multimodal Generative AI Systems Evaluation

Unit: COMP8410 Advanced Topics in Artificial Intelligence

Institution: Macquarie University

Author: Don Supun Nisala Gunasekara (Student ID: 49126873)

Role: Chief AI Officer (CAIO)

1. Project Overview

This repository contains the complete benchmarking harness, evaluation pipelines, datasets, and empirical findings for an Automated High-Fidelity Restaurant Marketing and Digital System. The project evaluates two contrasting multimodal generative AI pipelines across four distinct algorithmic stages (Weeks 2–7 curricula) to automate multi-channel restaurant marketing, WebGL interactive menus, and promotional video generation directly from unedited food photography.

2. The System Dualism

System A (Modular & Feed-Forward Baselines): Prioritizes throughput velocity, minimal compute footprints, and real-time operational dashboard integration.

System B (Attention-Guided & Spatiotemporal Latents): Prioritizes semantic isolation, structural identity preservation, and inter-frame temporal stability for broadcast and public-facing marketing.

3. Installation

Clone the repository and install the project dependencies:

git clone https://github.com/Supun222/COMP8410_A1.git
cd COMP8410_A1
pip install -r requirements.txt

Reproducing Benchmark Telemetry

Each pair can be executed independently from the repository root:

# Stage 1: 2D Image Editing

python src/run_pair1_evaluation.py

# Stage 2: 3D Generation

python src/run_pair2_evaluation.py

# Stage 3: Generative Video Synthesis

python src/run_pair3_evaluation.py

# Stage 4: Temporal Video Editing

python src/run_pair4_evaluation.py

All generated visual media will be saved to outputs/, and all comparative metric bar chart will be written to results/.
