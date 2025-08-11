# Physics-Informed Neural Network for Cooling Fin Heat Transfer

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://tensorflow.org/)

A Physics-Informed Neural Network (PINN) implementation for solving 1D transient heat transfer in aluminum cooling fins.

## Problem

Solves the 1D heat equation with convective losses:
```
ρcp ∂T/∂t = k ∂²T/∂x² - hP(T - T∞)
```

**Conditions:**
- Initial: T(x,0) = 373K (100°C)
- Base: T(0,t) = 373K (fixed)
- Tip: Convective cooling boundary

## Quick Start

```bash
pip install tensorflow numpy matplotlib seaborn scipy
python cooling_fin_pinn.py
```

## Results

- ✅ 5+ orders of magnitude loss reduction
- ✅ Physical temperature evolution (373K → 356K at tip)
- ✅ Smooth, continuous solution
- ✅ All boundary conditions satisfied

## Key Features

- **Meshless**: No grid discretization needed
- **Physics-embedded**: Heat equation built into loss function
- **Automatic differentiation**: TensorFlow handles gradients
- **Multiple visualizations**: Temperature evolution, heatmaps, 3D plots

## Network Architecture

```python
Input: [x, t] → Hidden: 4×64 neurons → Output: T(x,t)
Loss = PDE_residual + IC_loss + BC_loss
```

## Files Generated

- `cooling_fin_temperature_evolution.png` - Spatial temperature profiles
- `cooling_fin_heatmap.png` - 2D temperature field
- `cooling_fin_training_convergence.png` - Loss evolution
- `cooling_fin_temporal_evolution.png` - Time series at fixed positions
- `cooling_fin_3d_surface.png` - 3D visualization

## Applications

- Electronics cooling (CPU heat sinks)
- Automotive thermal systems
- Heat exchanger design
- Thermal optimization

## License

MIT