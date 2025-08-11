import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy.integrate import odeint
import time

np.random.seed(42)
tf.random.set_seed(42)

# Physical parameters
rho = 2700.0        # kg/m³ - density (aluminum)
c_p = 900.0         # J/(kg·K) - specific heat capacity (aluminum)
k = 200.0           # W/(m·K) - thermal conductivity (aluminum)
h = 50.0            # W/(m²·K) - heat transfer coefficient (convection)
P = 0.1             # m - perimeter
T_inf = 298.0       # K - ambient temperature
L = 1.0             # m - fin length
t_max = 1000.0      # s - simulation time 

# Initial and boundary temperatures
T_initial = 373.0       # K (100°C)
T_base = 373.0          # K (100°C) - fixed at base

# Thermal diffusivity and convective parameters
alpha = k / (rho * c_p)   # m²/s
beta = h * P / (rho * c_p) # 1/s

print(f"Thermal diffusivity α: {alpha:.2e} m²/s")
print(f"Convective parameter β: {beta:.2e} 1/s")

class CoolingFinPINN:
    def __init__(self, layers, lb, ub):
        """
        Physics-Informed Neural Network for cooling fin heat transfer.

        Args:
            layers: List defining network architecture [input_dim, hidden1, hidden2, ..., output_dim]
            lb: Lower bounds [x_min, t_min]
            ub: Upper bounds [x_max, t_max]
        """
        self.layers = layers
        self.lb = tf.constant(lb, dtype=tf.float32)
        self.ub = tf.constant(ub, dtype=tf.float32)

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
                tf.sqrt(2.0 / (self.layers[i] + self.layers[i+1])),
                trainable=True
            )
            b = tf.Variable(tf.zeros([self.layers[i+1]], dtype=tf.float32), trainable=True)
            weights.append(w)
            biases.append(b)
            
        return weights, biases
    
    def neural_net(self, X):
        """Forward pass through the neural network"""
        # Normalize inputs
        X_norm = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        
        H = X_norm
        for i in range(len(self.weights) - 1):
            H = tf.tanh(tf.add(tf.matmul(H, self.weights[i]), self.biases[i]))
        
        # Output layer (no activation)
        Y = tf.add(tf.matmul(H, self.weights[-1]), self.biases[-1])
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
        """Compute total loss"""
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
        
        # Total loss
        loss_total = loss_pde + loss_ic + loss_bc1 + loss_bc2
        
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
             x_bc2, t_bc2, epochs=10000, lr=0.001):
        """Train the PINN"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        print("Starting PINN training...")
        start_time = time.time()
        
        for epoch in range(epochs):
            loss_total, loss_pde, loss_ic, loss_bc = self.train_step(
                optimizer, x_pde, t_pde, x_ic, t_ic, T_ic, 
                x_bc1, t_bc1, T_bc1, x_bc2, t_bc2
            )
            
            # Store history
            self.loss_history.append(loss_total.numpy())
            self.pde_loss_history.append(loss_pde.numpy())
            self.ic_loss_history.append(loss_ic.numpy())
            self.bc_loss_history.append(loss_bc.numpy())
            
            if epoch % 1000 == 0:
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
    """Generate training data points"""
    # PDE collocation points
    n_pde = 5000
    x_pde = np.random.uniform(0, L, (n_pde, 1))
    t_pde = np.random.uniform(0, t_max, (n_pde, 1))
    
    # Initial condition points
    n_ic = 500
    x_ic = np.random.uniform(0, L, (n_ic, 1))
    t_ic = np.zeros((n_ic, 1))
    T_ic = T_initial * np.ones((n_ic, 1))
    
    # Boundary condition 1: x = 0
    n_bc1 = 250
    x_bc1 = np.zeros((n_bc1, 1))
    t_bc1 = np.random.uniform(0, t_max, (n_bc1, 1))
    T_bc1 = T_base * np.ones((n_bc1, 1))
    
    # Boundary condition 2: x = L
    n_bc2 = 250
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

def analytical_solution_approximation(x, t):
    """
    Approximate analytical solution using separation of variables
    For comparison purposes
    """
    # This is a simplified approximation for the cooling fin problem
    # In reality, the exact solution involves Bessel functions
    n_terms = 50
    result = np.zeros_like(x)
    
    for n in range(1, n_terms + 1):
        lambda_n = n * np.pi / L
        gamma_n = np.sqrt(lambda_n**2 + beta / alpha)
        
        # Coefficients (simplified)
        A_n = (2 / L) * (T_initial - T_inf) * np.sin(lambda_n * L) / lambda_n
        
        # Series term
        term = A_n * np.sin(lambda_n * x) * np.exp(-alpha * gamma_n**2 * t)
        result += term
    
    return result + T_inf

def create_visualizations(model):
    """Create comprehensive visualizations"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
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
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(t_test)))
    
    for i, t_val in enumerate(t_test):
        ax1.plot(x_test, T_pred[i, :], 'o-', color=colors[i], 
                linewidth=2, markersize=4, label=f't = {t_val} s')
    
    ax1.set_xlabel('Position x (m)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Temperature T (K)', fontsize=14, fontweight='bold')
    ax1.set_title('Temperature Distribution Along Cooling Fin\n(PINN Solution)', 
                  fontsize=16, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([295, 375])
    
    # Add temperature in Celsius on right axis
    ax1_celsius = ax1.twinx()
    ax1_celsius.set_ylabel('Temperature (°C)', fontsize=14, fontweight='bold')
    ax1_celsius.set_ylim([T-273.15 for T in ax1.get_ylim()])
    
    plt.tight_layout()
    plt.savefig('cooling_fin_temperature_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: 2D Heatmap
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Create finer grid for smooth visualization
    x_fine = np.linspace(0, L, 200)
    t_fine = np.linspace(0, t_max, 100)
    X_fine, T_fine = np.meshgrid(x_fine, t_fine)
    
    X_fine_flat = X_fine.flatten()[:, None]
    T_fine_flat = T_fine.flatten()[:, None]
    
    T_pred_fine = model.predict(
        tf.constant(X_fine_flat, dtype=tf.float32),
        tf.constant(T_fine_flat, dtype=tf.float32)
    ).numpy()
    T_pred_fine = T_pred_fine.reshape(X_fine.shape)
    
    im = ax2.contourf(X_fine, T_fine, T_pred_fine, levels=50, cmap='coolwarm')
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Temperature (K)', fontsize=14, fontweight='bold')
    
    # Add contour lines
    contours = ax2.contour(X_fine, T_fine, T_pred_fine, levels=15, colors='black', alpha=0.4, linewidths=0.5)
    ax2.clabel(contours, inline=True, fontsize=8, fmt='%.0f K')
    
    ax2.set_xlabel('Position x (m)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time t (s)', fontsize=14, fontweight='bold')
    ax2.set_title('Temperature Evolution Heatmap\n(PINN Solution)', 
                  fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cooling_fin_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 3: Training convergence
    fig3, ((ax31, ax32), (ax33, ax34)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(len(model.loss_history))
    
    # Total loss
    ax31.semilogy(epochs, model.loss_history, 'b-', linewidth=2)
    ax31.set_xlabel('Epoch')
    ax31.set_ylabel('Total Loss')
    ax31.set_title('Total Loss Convergence')
    ax31.grid(True, alpha=0.3)
    
    # PDE loss
    ax32.semilogy(epochs, model.pde_loss_history, 'r-', linewidth=2)
    ax32.set_xlabel('Epoch')
    ax32.set_ylabel('PDE Loss')
    ax32.set_title('Physics Loss Convergence')
    ax32.grid(True, alpha=0.3)
    
    # Initial condition loss
    ax33.semilogy(epochs, model.ic_loss_history, 'g-', linewidth=2)
    ax33.set_xlabel('Epoch')
    ax33.set_ylabel('Initial Condition Loss')
    ax33.set_title('IC Loss Convergence')
    ax33.grid(True, alpha=0.3)
    
    # Boundary condition loss
    ax34.semilogy(epochs, model.bc_loss_history, 'm-', linewidth=2)
    ax34.set_xlabel('Epoch')
    ax34.set_ylabel('Boundary Condition Loss')
    ax34.set_title('BC Loss Convergence')
    ax34.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cooling_fin_training_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 4: Temperature at specific locations vs time
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    
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
        
        ax4.plot(t_continuous, T_pred_pos.flatten(), color=colors[i], 
                linewidth=2, label=f'x = {pos:.2f} m')
    
    ax4.set_xlabel('Time t (s)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Temperature T (K)', fontsize=14, fontweight='bold')
    ax4.set_title('Temperature Evolution at Different Positions\n(PINN Solution)', 
                  fontsize=16, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add ambient temperature line
    ax4.axhline(y=T_inf, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Ambient T∞')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('cooling_fin_temporal_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig1, fig2, fig3, fig4

def analyze_heat_transfer_characteristics(model):
    """Analyze heat transfer characteristics"""
    print("\n" + "="*60)
    print("HEAT TRANSFER ANALYSIS")
    print("="*60)
    
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
    
    print(f"\nFinal temperature at tip (x=L): {T_final[-1, 0]:.1f} K ({T_final[-1, 0]-273.15:.1f}°C)")
    print(f"Temperature drop from base to tip: {T_base - T_final[-1, 0]:.1f} K")
    
    # Heat transfer rate at base
    with tf.GradientTape() as tape:
        x_base = tf.constant([[0.0]], dtype=tf.float32)
        t_base = tf.constant([[t_max]], dtype=tf.float32)
        tape.watch(x_base)
        X_base = tf.concat([x_base, t_base], axis=1)
        T_base_pred = model.neural_net(X_base)
    
    dT_dx_base = tape.gradient(T_base_pred, x_base)
    q_base = -k * dT_dx_base.numpy()[0, 0]  # Heat flux at base
    
    print(f"Heat flux at base: {q_base:.1f} W/m²")
    
    # Efficiency calculation (simplified)
    T_avg = np.mean(T_final)
    effectiveness = (T_avg - T_inf) / (T_base - T_inf)
    print(f"Fin effectiveness: {effectiveness:.3f}")

def main():
    """Main execution function"""
    print("Physics-Informed Neural Network for Cooling Fin Heat Transfer")
    print("="*65)
    
    # Network architecture
    layers = [2, 50, 50, 50, 50, 1]  # [input, hidden layers, output]
    
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
        epochs=15000, lr=0.001
    )
    
    # Create visualizations
    print("\nGenerating visualizations...")
    figs = create_visualizations(model)
    
    # Analyze results
    analyze_heat_transfer_characteristics(model)
    
    print(f"\nFinal training loss: {loss_history[-1]:.2e}")
    print("\nVisualization files saved:")
    print("  - cooling_fin_temperature_evolution.png")
    print("  - cooling_fin_heatmap.png") 
    print("  - cooling_fin_training_convergence.png")
    print("  - cooling_fin_temporal_evolution.png")
    
    return model, figs

if __name__ == "__main__":
    model, figures = main()