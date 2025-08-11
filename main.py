import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy.integrate import odeint
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Physical parameters
rho = 2700.0      # kg/m³ - density (aluminum)
c_p = 900.0       # J/(kg·K) - specific heat capacity
k = 200.0         # W/(m·K) - thermal conductivity
h = 50.0          # W/(m²·K) - convective heat transfer coefficient
P = 0.1           # m - perimeter
T_inf = 298.0     # K - ambient temperature (25°C)
L = 1.0           # m - fin length
t_max = 1000.0    # s - simulation time

# Initial and boundary temperatures
T_initial = 373.0  # K (100°C)
T_base = 373.0     # K (100°C) - fixed at base

# Thermal diffusivity and convective parameter
alpha = k / (rho * c_p)  # m²/s
beta = h * P / (rho * c_p)  # 1/s

print(f"Thermal diffusivity α = {alpha:.2e} m²/s")
print(f"Convective parameter β = {beta:.2e} 1/s")

class CoolingFinPINN:
    def __init__(self, layers, lb, ub):
        """
        Physics-Informed Neural Network for cooling fin heat transfer
        
        Args:
            layers: List defining network architecture [input_dim, hidden1, hidden2, ..., output_dim]
            lb: Lower bounds [x_min, t_min]
            ub: Upper bounds [x_max, t_max]
        """
        self.layers = layers
        self.lb = tf.constant(lb, dtype=tf.float32)
        self.ub = tf.constant(ub, dtype=tf.float32)
        
        # Temperature scaling for better training
        self.T_scale = (T_initial + T_inf) / 2.0
        self.T_range = T_initial - T_inf
        
        # Initialize network weights and biases
        self.weights, self.biases = self.initialize_network()
        
        # Training history
        self.loss_history = []
        self.pde_loss_history = []
        self.ic_loss_history = []
        self.bc_loss_history = []
        
    def initialize_network(self):
        """Initialize network parameters using Xavier initialization"""
        weights = []
        biases = []
        
        for i in range(len(self.layers) - 1):
            w = tf.Variable(
                tf.random.normal([self.layers[i], self.layers[i+1]], dtype=tf.float32) * 
                np.sqrt(2.0 / (self.layers[i] + self.layers[i+1])),
                trainable=True
            )
            b = tf.Variable(tf.zeros([self.layers[i+1]], dtype=tf.float32), trainable=True)
            weights.append(w)
            biases.append(b)
            
        return weights, biases
    
    def neural_net(self, X):
        """Forward pass through the neural network with proper scaling"""
        # Normalize inputs to [-1, 1]
        X_norm = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        
        H = X_norm
        for i in range(len(self.weights) - 1):
            H = tf.tanh(tf.add(tf.matmul(H, self.weights[i]), self.biases[i]))
        
        # Output layer with scaling to physical temperature range
        Y_raw = tf.add(tf.matmul(H, self.weights[-1]), self.biases[-1])
        
        # Scale output to temperature range [T_inf, T_initial]
        Y = T_inf + (T_initial - T_inf) * tf.sigmoid(Y_raw)
        
        return Y
    
    def physics_net(self, x, t):
        """Compute physics residuals"""
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, t])
                X = tf.concat([x, t], axis=1)
                T = self.neural_net(X)
            
            T_x = tape1.gradient(T, x)
            T_t = tape1.gradient(T, t)
        
        T_xx = tape2.gradient(T_x, x)
        del tape1, tape2
        
        # PDE residual: ρc_p ∂T/∂t = k ∂²T/∂x² - hP(T - T_∞)
        pde_residual = rho * c_p * T_t - k * T_xx + h * P * (T - T_inf)
        
        return pde_residual, T
    
    def get_trainable_variables(self):
        """Get all trainable variables"""
        return self.weights + self.biases
    
    @tf.function
    def loss_function(self, x_pde, t_pde, x_ic, t_ic, T_ic, 
                     x_bc1, t_bc1, T_bc1, x_bc2, t_bc2):
        """Compute total loss with proper weighting"""
        # PDE loss
        pde_residual, _ = self.physics_net(x_pde, t_pde)
        loss_pde = tf.reduce_mean(tf.square(pde_residual))
        
        # Initial condition loss
        X_ic = tf.concat([x_ic, t_ic], axis=1)
        T_pred_ic = self.neural_net(X_ic)
        loss_ic = tf.reduce_mean(tf.square(T_pred_ic - T_ic))
        
        # Boundary condition 1: T(0,t) = T_base
        X_bc1 = tf.concat([x_bc1, t_bc1], axis=1)
        T_pred_bc1 = self.neural_net(X_bc1)
        loss_bc1 = tf.reduce_mean(tf.square(T_pred_bc1 - T_bc1))
        
        # Boundary condition 2: k ∂T/∂x|_{x=L} = -h(T(L,t) - T_∞)
        with tf.GradientTape() as tape:
            tape.watch(x_bc2)
            X_bc2 = tf.concat([x_bc2, t_bc2], axis=1)
            T_pred_bc2 = self.neural_net(X_bc2)
        T_x_bc2 = tape.gradient(T_pred_bc2, x_bc2)
        
        bc2_residual = k * T_x_bc2 + h * (T_pred_bc2 - T_inf)
        loss_bc2 = tf.reduce_mean(tf.square(bc2_residual))
        
        # Weighted total loss for better convergence
        w_pde = 1e-8   # Scale down PDE loss due to large thermal values
        w_ic = 1.0     # Keep IC loss at normal scale
        w_bc = 1.0     # Keep BC loss at normal scale
        
        loss_total = w_pde * loss_pde + w_ic * loss_ic + w_bc * (loss_bc1 + loss_bc2)
        
        return loss_total, loss_pde, loss_ic, (loss_bc1 + loss_bc2)
    
    def train_step(self, optimizer, x_pde, t_pde, x_ic, t_ic, T_ic,
                  x_bc1, t_bc1, T_bc1, x_bc2, t_bc2):
        """Single training step"""
        with tf.GradientTape() as tape:
            loss_total, loss_pde, loss_ic, loss_bc = self.loss_function(
                x_pde, t_pde, x_ic, t_ic, T_ic, x_bc1, t_bc1, T_bc1, x_bc2, t_bc2
            )
        
        gradients = tape.gradient(loss_total, self.get_trainable_variables())
        optimizer.apply_gradients(zip(gradients, self.get_trainable_variables()))
        
        return loss_total, loss_pde, loss_ic, loss_bc
    
    def train(self, x_pde, t_pde, x_ic, t_ic, T_ic, x_bc1, t_bc1, T_bc1, 
             x_bc2, t_bc2, epochs=20000, lr=0.001):
        """Train the PINN with adaptive learning rate"""
        # Use learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        print("Starting PINN training...")
        start_time = time.time()
        
        for epoch in range(epochs):
            loss_total, loss_pde, loss_ic, loss_bc = self.train_step(
                optimizer, x_pde, t_pde, x_ic, t_ic, T_ic, 
                x_bc1, t_bc1, T_bc1, x_bc2, t_bc2
            )
            
            # Store history
            self.loss_history.append(float(loss_total.numpy()))
            self.pde_loss_history.append(float(loss_pde.numpy()))
            self.ic_loss_history.append(float(loss_ic.numpy()))
            self.bc_loss_history.append(float(loss_bc.numpy()))
            
            if epoch % 2000 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:5d}: Loss = {loss_total:.2e}, "
                      f"PDE = {loss_pde:.2e}, IC = {loss_ic:.2e}, "
                      f"BC = {loss_bc:.2e}, Time = {elapsed:.1f}s")
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        return self.loss_history
    
    def predict(self, x, t):
        """Make predictions"""
        X = tf.concat([x, t], axis=1)
        return self.neural_net(X)

def generate_training_data():
    """Generate training data points with better distribution"""
    # PDE collocation points
    n_pde = 10000
    x_pde = np.random.uniform(0, L, (n_pde, 1))
    t_pde = np.random.uniform(0, t_max, (n_pde, 1))
    
    # Initial condition points
    n_ic = 1000
    x_ic = np.random.uniform(0, L, (n_ic, 1))
    t_ic = np.zeros((n_ic, 1))
    T_ic = T_initial * np.ones((n_ic, 1))
    
    # Boundary condition 1: x = 0
    n_bc1 = 500
    x_bc1 = np.zeros((n_bc1, 1))
    t_bc1 = np.random.uniform(0, t_max, (n_bc1, 1))
    T_bc1 = T_base * np.ones((n_bc1, 1))
    
    # Boundary condition 2: x = L
    n_bc2 = 500
    x_bc2 = L * np.ones((n_bc2, 1))
    t_bc2 = np.random.uniform(0, t_max, (n_bc2, 1))
    
    # Convert to tensors
    data = {
        'x_pde': tf.constant(x_pde, dtype=tf.float32),
        't_pde': tf.constant(t_pde, dtype=tf.float32),
        'x_ic': tf.constant(x_ic, dtype=tf.float32),
        't_ic': tf.constant(t_ic, dtype=tf.float32),
        'T_ic': tf.constant(T_ic, dtype=tf.float32),
        'x_bc1': tf.constant(x_bc1, dtype=tf.float32),
        't_bc1': tf.constant(t_bc1, dtype=tf.float32),
        'T_bc1': tf.constant(T_bc1, dtype=tf.float32),
        'x_bc2': tf.constant(x_bc2, dtype=tf.float32),
        't_bc2': tf.constant(t_bc2, dtype=tf.float32),
    }
    
    print(f"Generated training data:")
    print(f"  PDE points: {n_pde}")
    print(f"  IC points: {n_ic}")
    print(f"  BC points: {n_bc1 + n_bc2}")
    
    return data

def create_visualizations(model):
    """Create comprehensive visualizations with proper scaling"""
    
    # Create prediction grid
    x_test = np.linspace(0, L, 101)
    t_test = np.array([0, 50, 100, 200, 400, 600, 800, 1000])
    
    X_test, T_test = np.meshgrid(x_test, t_test)
    X_flat = X_test.flatten()[:, None]
    T_flat = T_test.flatten()[:, None]
    
    # PINN predictions
    T_pred = model.predict(
        tf.constant(X_flat, dtype=tf.float32),
        tf.constant(T_flat, dtype=tf.float32)
    ).numpy()
    T_pred = T_pred.reshape(X_test.shape)
    
    # Figure 1: Temperature evolution over time
    plt.figure(figsize=(14, 10))
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(t_test)))
    
    for i, t_val in enumerate(t_test):
        plt.plot(x_test, T_pred[i, :], 'o-', color=colors[i], 
                linewidth=3, markersize=6, label=f't = {t_val} s', alpha=0.8)
    
    plt.xlabel('Position x (m)', fontsize=16, fontweight='bold')
    plt.ylabel('Temperature T (K)', fontsize=16, fontweight='bold')
    plt.title('Temperature Distribution Along Cooling Fin\n(PINN Solution)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim([290, 380])
    
    # Add temperature in Celsius on right axis
    ax1 = plt.gca()
    ax1_celsius = ax1.twinx()
    ax1_celsius.set_ylabel('Temperature (°C)', fontsize=16, fontweight='bold')
    ax1_celsius.set_ylim([T-273.15 for T in ax1.get_ylim()])
    
    plt.tight_layout()
    plt.savefig('cooling_fin_temperature_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: 2D Heatmap
    plt.figure(figsize=(14, 10))
    
    # Create finer grid for smooth visualization
    x_fine = np.linspace(0, L, 200)
    t_fine = np.linspace(0, t_max, 150)
    X_fine, T_fine = np.meshgrid(x_fine, t_fine)
    
    X_fine_flat = X_fine.flatten()[:, None]
    T_fine_flat = T_fine.flatten()[:, None]
    
    T_pred_fine = model.predict(
        tf.constant(X_fine_flat, dtype=tf.float32),
        tf.constant(T_fine_flat, dtype=tf.float32)
    ).numpy()
    T_pred_fine = T_pred_fine.reshape(X_fine.shape)
    
    # Create contour plot
    levels = np.linspace(T_inf, T_initial, 20)
    im = plt.contourf(X_fine, T_fine, T_pred_fine, levels=levels, cmap='coolwarm', extend='both')
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Temperature (K)', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Add contour lines
    contours = plt.contour(X_fine, T_fine, T_pred_fine, levels=10, colors='black', alpha=0.4, linewidths=1)
    plt.clabel(contours, inline=True, fontsize=10, fmt='%.0f K')
    
    plt.xlabel('Position x (m)', fontsize=16, fontweight='bold')
    plt.ylabel('Time t (s)', fontsize=16, fontweight='bold')
    plt.title('Temperature Evolution Heatmap\n(PINN Solution)', 
              fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('cooling_fin_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 3: Training convergence
    fig, ((ax31, ax32), (ax33, ax34)) = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(len(model.loss_history))
    
    # Total loss
    ax31.semilogy(epochs, model.loss_history, 'b-', linewidth=2)
    ax31.set_xlabel('Epoch', fontsize=14)
    ax31.set_ylabel('Total Loss', fontsize=14)
    ax31.set_title('Total Loss Convergence', fontsize=16, fontweight='bold')
    ax31.grid(True, alpha=0.3)
    
    # PDE loss
    ax32.semilogy(epochs, model.pde_loss_history, 'r-', linewidth=2)
    ax32.set_xlabel('Epoch', fontsize=14)
    ax32.set_ylabel('PDE Loss', fontsize=14)
    ax32.set_title('Physics Loss Convergence', fontsize=16, fontweight='bold')
    ax32.grid(True, alpha=0.3)
    
    # Initial condition loss
    ax33.semilogy(epochs, model.ic_loss_history, 'g-', linewidth=2)
    ax33.set_xlabel('Epoch', fontsize=14)
    ax33.set_ylabel('Initial Condition Loss', fontsize=14)
    ax33.set_title('IC Loss Convergence', fontsize=16, fontweight='bold')
    ax33.grid(True, alpha=0.3)
    
    # Boundary condition loss
    ax34.semilogy(epochs, model.bc_loss_history, 'm-', linewidth=2)
    ax34.set_xlabel('Epoch', fontsize=14)
    ax34.set_ylabel('Boundary Condition Loss', fontsize=14)
    ax34.set_title('BC Loss Convergence', fontsize=16, fontweight='bold')
    ax34.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cooling_fin_training_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 4: Temperature at specific locations vs time
    plt.figure(figsize=(14, 10))
    
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]  # Different positions along the fin
    t_continuous = np.linspace(0, t_max, 1000)
    colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
    
    for i, pos in enumerate(positions):
        x_pos = pos * np.ones((len(t_continuous), 1))
        t_pos = t_continuous.reshape(-1, 1)
        
        T_pred_pos = model.predict(
            tf.constant(x_pos, dtype=tf.float32),
            tf.constant(t_pos, dtype=tf.float32)
        ).numpy()
        
        plt.plot(t_continuous, T_pred_pos.flatten(), color=colors[i], 
                linewidth=3, label=f'x = {pos:.2f} m')
    
    plt.xlabel('Time t (s)', fontsize=16, fontweight='bold')
    plt.ylabel('Temperature T (K)', fontsize=16, fontweight='bold')
    plt.title('Temperature Evolution at Different Positions\n(PINN Solution)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add ambient temperature line
    plt.axhline(y=T_inf, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Ambient T∞')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('cooling_fin_temporal_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 5: 3D Surface Plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh for 3D plot
    x_3d = np.linspace(0, L, 50)
    t_3d = np.linspace(0, t_max, 50)
    X_3d, T_3d = np.meshgrid(x_3d, t_3d)
    
    X_3d_flat = X_3d.flatten()[:, None]
    T_3d_flat = T_3d.flatten()[:, None]
    
    T_pred_3d = model.predict(
        tf.constant(X_3d_flat, dtype=tf.float32),
        tf.constant(T_3d_flat, dtype=tf.float32)
    ).numpy()
    T_pred_3d = T_pred_3d.reshape(X_3d.shape)
    
    surf = ax.plot_surface(X_3d, T_3d, T_pred_3d, cmap='coolwarm', alpha=0.8)
    
    ax.set_xlabel('Position x (m)', fontsize=14)
    ax.set_ylabel('Time t (s)', fontsize=14)
    ax.set_zlabel('Temperature T (K)', fontsize=14)
    ax.set_title('3D Temperature Distribution\n(PINN Solution)', fontsize=16, fontweight='bold')
    
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
    cbar.set_label('Temperature (K)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('cooling_fin_3d_surface.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_heat_transfer_characteristics(model):
    """Analyze heat transfer characteristics"""
    print("\n" + "="*70)
    print("HEAT TRANSFER ANALYSIS RESULTS")
    print("="*70)
    
    # Time constants
    tau_conv = rho * c_p / (h * P)  # Convective time constant
    tau_diff = L**2 / alpha          # Diffusive time constant
    
    print(f"Convective time constant τ_conv = {tau_conv:.1f} s")
    print(f"Diffusive time constant τ_diff = {tau_diff:.1f} s")
    
    # Biot number
    Bi = h * L / k
    print(f"Biot number Bi = {Bi:.4f}")
    
    if Bi < 0.1:
        print("  → Lumped capacitance model would be adequate")
    else:
        print("  → Distributed model necessary (temperature gradients significant)")
    
    # Fourier number at t_max
    Fo = alpha * t_max / L**2
    print(f"Fourier number at t_max: Fo = {Fo:.2f}")
    
    # Predict final temperature distribution
    x_final = np.linspace(0, L, 101).reshape(-1, 1)
    t_final = t_max * np.ones_like(x_final)
    
    T_final = model.predict(
        tf.constant(x_final, dtype=tf.float32),
        tf.constant(t_final, dtype=tf.float32)
    ).numpy()
    
    print(f"\nTemperature Analysis at t = {t_max} s:")
    print(f"Final temperature at base (x=0): {T_final[0, 0]:.1f} K ({T_final[0, 0]-273.15:.1f}°C)")
    print(f"Final temperature at tip (x=L): {T_final[-1, 0]:.1f} K ({T_final[-1, 0]-273.15:.1f}°C)")
    print(f"Temperature drop from base to tip: {T_final[0, 0] - T_final[-1, 0]:.1f} K")
    
    # Average temperature
    T_avg = np.mean(T_final)
    print(f"Average fin temperature: {T_avg:.1f} K ({T_avg-273.15:.1f}°C)")
    
    # Fin effectiveness
    effectiveness = (T_avg - T_inf) / (T_base - T_inf)
    print(f"Fin effectiveness: {effectiveness:.3f}")
    
    # Heat transfer rate at base (approximate)
    x_base_grad = np.array([[0.001], [t_max]])  # Small perturbation for gradient
    with tf.GradientTape() as tape:
        x_tensor = tf.Variable([[0.0]], dtype=tf.float32)
        t_tensor = tf.constant([[t_max]], dtype=tf.float32)
        X_base = tf.concat([x_tensor, t_tensor], axis=1)
        T_base_pred = model.neural_net(X_base)
    
    print(f"\nFin Performance Summary:")
    print(f"  Initial temperature: {T_initial:.1f} K ({T_initial-273.15:.1f}°C)")
    print(f"  Ambient temperature: {T_inf:.1f} K ({T_inf-273.15:.1f}°C)")
    print(f"  Temperature difference: {T_initial - T_inf:.1f} K")
    print(f"  Final average cooling: {T_initial - T_avg:.1f} K")
    print(f"  Cooling efficiency: {(T_initial - T_avg)/(T_initial - T_inf)*100:.1f}%")

def main():
    """Main execution function"""
    print("Physics-Informed Neural Network for Cooling Fin Heat Transfer")
    print("="*70)
    
    # Network architecture - deeper network for better approximation
    layers = [2, 64, 64, 64, 64, 1]  # [input, hidden layers, output]
    
    # Domain bounds
    lb = [0.0, 0.0]      # [x_min, t_min]
    ub = [L, t_max]      # [x_max, t_max]
    
    # Generate training data
    training_data = generate_training_data()
    
    # Initialize and train PINN
    print(f"\nInitializing PINN with architecture: {layers}")
    model = CoolingFinPINN(layers, lb, ub)
    
    # Train the model
    loss_history = model.train(
        training_data['x_pde'], training_data['t_pde'],
        training_data['x_ic'], training_data['t_ic'], training_data['T_ic'],
        training_data['x_bc1'], training_data['t_bc1'], training_data['T_bc1'],
        training_data['x_bc2'], training_data['t_bc2'],
        epochs=20000, lr=0.001
    )
    
    # Create visualizations
    print("\nGenerating comprehensive visualizations...")
    create_visualizations(model)
    
    # Analyze results
    analyze_heat_transfer_characteristics(model)
    
    print(f"\nFinal training loss: {loss_history[-1]:.2e}")
    print("\nVisualization files saved:")
    print("  - cooling_fin_temperature_evolution.png")
    print("  - cooling_fin_heatmap.png") 
    print("  - cooling_fin_training_convergence.png")
    print("  - cooling_fin_temporal_evolution.png")
    print("  - cooling_fin_3d_surface.png")
    
    return model

if __name__ == "__main__":
    model = main()