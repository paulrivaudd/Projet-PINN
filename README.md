# 🧠 Solving PDEs with Neural Networks — PINNs & Deep Ritz

> Mesh-free numerical resolution of **partial differential equations (PDEs)** using neural networks, in **PyTorch**.
> First-year engineering project — *École nationale des ponts et chaussées* (CERMICS laboratory).

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-Beamer-008080?logo=latex&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

Classical PDE solvers (finite differences, finite elements) rely on **meshing** the domain — which becomes expensive in high dimension and awkward for complex geometries. This project explores a modern, **mesh-free** alternative: approximating the solution **directly with a neural network**.

Two paradigms are implemented and compared in PyTorch:

- **PINN — Physics-Informed Neural Network** *(strong form)*: the network minimizes the **residual** of the PDE plus boundary-condition penalties.
- **Deep Ritz** *(weak / variational form)*: the network minimizes an **energy functional**.

**Test problems** (1D, on the open interval $(0,1)$): the **Poisson** equation $-u'' = f$ and a **modified Helmholtz** equation $-\Delta u + \lambda u = f$.

## ✨ Highlights

- **Exact** network derivatives via **automatic differentiation** (`torch.func.grad` composed twice to get $u''$) — no finite-difference step or truncation error.
- Fully **vectorized** collocation with **`vmap`** (no Python loops).
- Systematic study of **activation functions** and **architecture** (width & depth), with a **validation set** and **best-model checkpointing**.
- Clean, honest **PINN vs Deep Ritz** comparison (cost, accuracy, applicability).

## 📊 Key results — 1D Poisson, $f(x)=\sin(2\pi x)$

| Method  | Activation | Neurons | Layers | Min. validation loss |
|---------|------------|--------:|-------:|----------------------|
| PINN    | tanh       |       3 |      1 | 5.1×10⁻³             |
| PINN    | tanh       |      10 |      1 | **1.7×10⁻⁵**         |
| PINN    | sigmoid    |      50 |      1 | 1.6×10⁻⁵             |
| PINN    | ReLU       |      50 |      1 | 5.0×10⁻¹ *(fails)*   |
| Deep NN | tanh       |       5 |     20 | 4.8×10⁻¹ *(unstable)*|

**Takeaways.** `tanh` is the right choice (smooth ⇒ $u''$ well-defined and non-zero); **10 neurons are enough** in 1D; **ReLU fails** (zero second derivative ⇒ the residual cannot be minimized); networks deeper than ~17 layers collapse (**vanishing gradient**). Deep Ritz is slower than PINN on Poisson but needs only first derivatives and handles the Helmholtz case well ($L^2 \approx 10^{-5}$).

## 🧩 Method at a glance

The PINN loss, discretized on $N$ collocation points $x_i$:

$$\mathcal{L}(\theta)=\underbrace{\frac{1}{N}\sum_{i=1}^{N}\bigl(u_\theta''(x_i)+f(x_i)\bigr)^2}_{\text{PDE residual}}+\underbrace{\bigl(u_\theta(0)-u_0\bigr)^2+\bigl(u_\theta(1)-u_1\bigr)^2}_{\text{boundary conditions}}$$

Training minimizes $\mathcal{L}$ with **Adam**; the exact solution is the one that drives this energy to zero.

## 📁 Repository structure

```text
pinn-deepritz-pde/
├── README.md
├── LICENSE
├── .gitignore
├── code/
│   └── PROJET.ipynb                     # PyTorch implementation (PINN & Deep Ritz)
├── report/
│   └── rapport.pdf                      # Full project report (FR)
├── slides/
│   ├── Partie1_PINN_slides.tex          # Beamer slides — Part 1 (intro & PINN method)
│   └── Partie1_PINN_slides.pdf
└── docs/
    ├── Partie1_PINN_script_oral.tex     # Oral presentation script (FR)
    ├── Partie1_PINN_script_oral.pdf
    ├── Partie1_PINN_fiche_revision.tex  # Study / revision sheet (FR)
    └── Partie1_PINN_fiche_revision.pdf
```

## 🚀 Getting started

**Run the code**
```bash
pip install torch numpy matplotlib seaborn
jupyter notebook code/PROJET.ipynb
```

**Build the slides / documents** (requires a TeX distribution with Beamer + Latin Modern)
```bash
cd slides && pdflatex Partie1_PINN_slides.tex && pdflatex Partie1_PINN_slides.tex
```


## 🛠 Tech stack

Python · PyTorch (`torch.func`: `grad`, `vmap`) · NumPy · Matplotlib · Seaborn · LaTeX / Beamer

## 👥 Authors & context

- **Authors:** Paul Rivaud, Timoté Missaye, Soufiane Ait Abbou, Pierre Kairo
- **Supervisor:** Sofiane Ezzehi (CERMICS)
- **Institution:** École nationale des ponts et chaussées — first-year project, 2026
- *Project documents (report, slides, script, revision sheet) are written in French.*

## 📚 References

1. G. Cybenko (1989). *Approximation by superpositions of a sigmoidal function.*
2. W. E & B. Yu (2018). *The Deep Ritz Method.*
3. M. Raissi, P. Perdikaris & G. E. Karniadakis (2019). *Physics-Informed Neural Networks.* J. Comput. Phys. 378.

## 📄 License

Released under the MIT License — see [`LICENSE`](LICENSE). As this is a group project, please confirm with the co-authors before reuse.
