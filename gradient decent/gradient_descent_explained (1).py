"""
================================================================================
  GRADIENT DESCENT & COST FUNCTION — COMPLETE GUIDE WITH EXPLANATIONS
  Based on CampusX 100 Days of Machine Learning — Day 51
================================================================================

WHAT IS GRADIENT DESCENT?
--------------------------
Gradient Descent is an OPTIMIZATION ALGORITHM used to minimize the cost/loss
function of a machine learning model. It finds the best parameters (m, b) for
a linear model by iteratively moving in the direction that reduces error.

Analogy: Imagine you're blindfolded on a hilly terrain and want to reach the
lowest valley. You feel the slope under your feet and take small steps in the
downhill direction. That's exactly what gradient descent does — it feels the
"slope" of the cost function and steps toward the minimum.

WHAT IS THE COST FUNCTION?
---------------------------
For Linear Regression: y = m*x + b
  - m = slope (weight/coefficient)
  - b = intercept (bias)

Cost Function (Sum of Squared Errors):
  Cost(m, b) = Σ (y_actual - y_predicted)²
             = Σ (y - (m*x + b))²

Goal: Find m and b that MINIMIZE this cost.

KEY CONCEPTS:
  - Learning Rate (lr): How big each step is. Too big → overshoot. Too small → slow.
  - Epochs: Number of full passes through all training data.
  - Slope/Gradient: Direction and magnitude of the hill at current position.
  - Convergence: When the cost stops decreasing significantly (reached the valley).

GRADIENT FORMULAS (derivatives of cost w.r.t. b and m):
  ∂Cost/∂b = -2 * Σ (y - (m*x + b))       ← slope w.r.t. intercept
  ∂Cost/∂m = -2 * Σ (y - (m*x + b)) * x   ← slope w.r.t. slope

UPDATE RULE:
  b_new = b_old - (learning_rate * ∂Cost/∂b)
  m_new = m_old - (learning_rate * ∂Cost/∂m)
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS & SETUP
# ==============================================================================

from sklearn.datasets import make_regression  # Generates synthetic regression data
import numpy as np                            # Numerical computing (arrays, math ops)
import matplotlib.pyplot as plt               # 2D plotting
import plotly.graph_objects as go             # Interactive 3D plots
import plotly.express as px                   # High-level interactive plots
import matplotlib.animation as animation      # Animated plots
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression  # sklearn's built-in linear regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ==============================================================================
# SECTION 2: GENERATE SYNTHETIC REGRESSION DATA
# ==============================================================================
"""
make_regression() creates a simple dataset for testing regression algorithms.

Parameters:
  n_samples=100   → 100 data points (rows)
  n_features=1    → 1 input feature (1 column X)
  n_informative=1 → All features are useful (no noise features)
  n_targets=1     → Single output value y
  noise=20        → Adds random noise (standard deviation=20) to make it realistic
  random_state=13 → Seed for reproducibility — same data every run

Output:
  X → shape (100, 1), input features
  y → shape (100,),   target values (what we want to predict)
"""
X, y = make_regression(
    n_samples=100,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=20,
    random_state=13
)

# Quick visualization to understand the data
plt.scatter(X, y)
plt.title("Synthetic Regression Data")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.show()


# ==============================================================================
# SECTION 3: VISUALIZE THE 3D COST FUNCTION SURFACE
# ==============================================================================
"""
Before running gradient descent, let's VISUALIZE what the cost function looks
like as a 3D surface. This helps us understand what "minimizing cost" means
geometrically.

We create a GRID of (m, b) values and compute cost for each combination.
The resulting 3D bowl-shaped surface shows:
  - X-axis: slope values (m)
  - Y-axis: intercept values (b)
  - Z-axis: cost at each (m, b) pair
  
The LOWEST POINT of this bowl = optimal (m, b) = best fit line.
"""

# Create a range of 10 evenly spaced values for m (slope) and b (intercept)
# linspace(start, stop, num) → array of 'num' values from start to stop
m_arr = np.linspace(-150, 150, 10)   # 10 slope values: [-150, ..., 150]
b_arr = np.linspace(-150, 150, 10)   # 10 intercept values: [-150, ..., 150]

# Create a 2D grid from m_arr and b_arr
# meshgrid turns two 1D arrays into two 2D arrays covering all combinations
# mGrid[i,j] = m_arr[j], bGrid[i,j] = b_arr[i]  → 10x10 = 100 combinations
mGrid, bGrid = np.meshgrid(m_arr, b_arr)

# Flatten the grids and stack them into a (100, 2) array
# Each row is one (m, b) pair to evaluate cost for
# .ravel() → flattens 2D to 1D
# .reshape(1, 100) → makes it a row vector
# np.vstack → stacks two row vectors vertically → shape (2, 100)
# .T → transpose → shape (100, 2)
final = np.vstack((
    mGrid.ravel().reshape(1, 100),
    bGrid.ravel().reshape(1, 100)
)).T
# final[i] = [m_value, b_value] for the i-th combination

# Compute the cost for each (m, b) pair
z_arr = []
for i in range(final.shape[0]):        # Loop over all 100 (m, b) combinations
    m_val = final[i, 0]                # Extract slope for this iteration
    b_val = final[i, 1]                # Extract intercept for this iteration
    
    # Cost = Sum of Squared Errors over all 100 data points
    # y - m*x - b  → residual (prediction error) for each point
    # **2          → square the error (makes it positive, penalizes large errors more)
    # np.sum(...)  → add up all squared errors
    cost = np.sum((y - m_val * X.reshape(100) - b_val) ** 2)
    z_arr.append(cost)

# Reshape cost array to 10x10 (matching our grid dimensions)
z_arr = np.array(z_arr).reshape(10, 10)

# Plot the 3D cost surface using Plotly (interactive — you can rotate it!)
fig = go.Figure(data=[go.Surface(x=m_arr, y=b_arr, z=z_arr)])
fig.update_layout(
    title='Cost Function Surface',
    autosize=False,
    width=500,
    height=500,
    margin=dict(l=65, r=50, b=65, t=90)
)
fig.show()
fig.write_html("cost_function.html")   # Save as interactive HTML file


# ==============================================================================
# SECTION 4: GRADIENT DESCENT — UPDATING BOTH m AND b
# ==============================================================================
"""
Now we actually RUN gradient descent. We start with arbitrary (bad) values of
m and b, and iteratively improve them using the gradient formulas.

ALGORITHM:
  For each epoch:
    1. Compute the gradient (slope of cost) w.r.t. b and m
    2. Update b and m by stepping opposite to the gradient
    3. Record the new values and cost for visualization

Initial values are deliberately "wrong" to show how GD corrects them.
"""

b = 150          # Starting intercept (far from optimal, which is ~-2)
m = -127.82      # Starting slope (far from optimal, which is ~28)
lr = 0.001       # Learning rate — controls step size
                 # Too high → oscillates, diverges | Too low → very slow convergence

all_b = []       # Track intercept at each epoch (to visualize convergence)
all_m = []       # Track slope at each epoch
all_cost = []    # Track cost at each epoch

epochs = 30      # Number of complete passes through all training data
                 # More epochs = more refinement, but diminishing returns after convergence

for i in range(epochs):
    slope_b = 0  # Gradient accumulator for intercept b
    slope_m = 0  # Gradient accumulator for slope m
    cost = 0     # Cost accumulator for this epoch

    # Inner loop: go through EVERY data point (this is Batch Gradient Descent)
    for j in range(X.shape[0]):   # X.shape[0] = 100 data points
        
        # GRADIENT for intercept b:
        # ∂Cost/∂b = -2 * (y - m*x - b) summed over all points
        # We accumulate the sum here; the -2 makes it negative → we negate in update
        slope_b = slope_b - 2 * (y[j] - (m * X[j]) - b)
        
        # GRADIENT for slope m:
        # ∂Cost/∂m = -2 * (y - m*x - b) * x  ← note the extra x multiplication
        # x acts as a "weight" — features with larger x have bigger influence
        slope_m = slope_m - 2 * (y[j] - (m * X[j]) - b) * X[j]
        
        # COST for this data point:
        # (y - predicted_y)² = squared error for one point
        cost = cost + (y[j] - m * X[j] - b) ** 2

    # UPDATE RULE — move opposite to gradient direction
    # If gradient is positive (cost goes up as b increases) → decrease b
    # If gradient is negative (cost goes down as b increases) → increase b
    b = b - (lr * slope_b)   # New intercept
    m = m - (lr * slope_m)   # New slope

    # Record history for visualization
    all_b.append(b)
    all_m.append(m)
    all_cost.append(cost)

print(f"Final m (slope):     {m:.4f}  (sklearn would give ~28)")
print(f"Final b (intercept): {b:.4f}  (sklearn would give ~-2)")


# ==============================================================================
# SECTION 5: VISUALIZE GRADIENT DESCENT PATH ON 3D SURFACE
# ==============================================================================
"""
Now we overlay the path taken by gradient descent ON the cost function surface.
Each dot = one epoch's (m, b, cost) position.
You'll see the path spiraling DOWN from a high point to the bowl's minimum!
"""

fig = px.scatter_3d(
    x=np.array(all_m).ravel(),       # m values at each epoch
    y=np.array(all_b).ravel(),       # b values at each epoch
    z=np.array(all_cost).ravel() * 100  # cost * 100 for visual scaling
)

# Add the cost function surface underneath the path
fig.add_trace(go.Surface(x=m_arr, y=b_arr, z=z_arr * 100))
fig.show()
fig.write_html("cost_function2.html")   # Save interactive HTML


# ==============================================================================
# SECTION 6: CONTOUR PLOT — 2D BIRD'S-EYE VIEW OF COST FUNCTION
# ==============================================================================
"""
A CONTOUR PLOT is a 2D top-down view of the 3D cost surface.
Think of it like a topographic map:
  - Outer rings = higher cost (hills)
  - Inner rings = lower cost (valleys)
  - The center = minimum (optimal m and b)
  
The WHITE LINE traces the gradient descent path across these contours.
You'll see it moving from outer rings toward the center.
"""

# Plotly interactive contour + gradient descent path
fig = go.Figure(go.Scatter(
    x=np.array(all_m).ravel(),    # m values (x-axis)
    y=np.array(all_b).ravel(),    # b values (y-axis)
    name='Gradient Descent Path',
    line=dict(color='#fff', width=4)  # White line for visibility
))

fig.add_trace(go.Contour(z=z_arr, x=m_arr, y=b_arr))  # Cost contours in background
fig.show()

# Matplotlib static version (more customizable, good for papers/reports)
fig, ax = plt.subplots(1, 1)
plt.figure(figsize=(18, 4))

# contourf = filled contour plot (colors filled between contour lines)
cp = ax.contourf(m_arr, b_arr, z_arr)
ax.plot(
    np.array(all_m).ravel(),
    np.array(all_b).ravel(),
    color='white'                  # GD path shown in white
)
fig.colorbar(cp)                   # Legend showing cost values for each color
ax.set_title('Cost Function Contour Map — Gradient Descent Path')
ax.set_xlabel('m (slope)')
ax.set_ylabel('b (intercept)')
plt.show()


# ==============================================================================
# SECTION 7: ANIMATED CONTOUR — WATCH GRADIENT DESCENT MOVE STEP BY STEP
# ==============================================================================
"""
This animation shows gradient descent moving across the contour map,
one epoch at a time. You literally see it "walking downhill"!

FuncAnimation works by calling animate(i) for each frame i = 0, 1, 2, ..., 29
Each frame adds the NEXT epoch's (m, b) position to the path.
"""

# %matplotlib notebook   # Uncomment in Jupyter to enable interactive animation

num_epochs = list(range(0, 30))   # List [0, 1, 2, ..., 29] for frame indices

fig = plt.figure(figsize=(9, 5))
axis = plt.axes(xlim=(-150, 150), ylim=(-150, 150))  # Set axis limits to match our search range

# Draw the static contour map in the background
axis.contourf(m_arr, b_arr, z_arr)

# Create an empty line that will grow with each animation frame
line, = axis.plot([], [], lw=2, color='white')

xdata, ydata = [], []   # Lists that accumulate m and b values across frames

def animate_contour(i):
    """
    Called once per frame. Adds the i-th epoch's (m, b) to the path.
    
    Parameters:
      i → current frame number (0 to 29)
    """
    label = f'epoch {i + 1}'          # Human-readable epoch label
    xdata.append(all_m[i])            # Add this epoch's slope to path
    ydata.append(all_b[i])            # Add this epoch's intercept to path
    line.set_data(xdata, ydata)       # Update the line with new data
    axis.set_xlabel(label)            # Show current epoch on x-axis label
    return line,

anim = animation.FuncAnimation(
    fig,
    animate_contour,
    frames=30,          # 30 frames = 30 epochs
    repeat=False,       # Don't loop the animation
    interval=500        # 500ms = 0.5 seconds between frames
)
# Uncomment to save as gif:
# writergif = animation.PillowWriter(fps=2)
# anim.save("animation8.gif", writer=writergif)


# ==============================================================================
# SECTION 8: FULL DEMO — GRADIENT DESCENT WITH DIFFERENT STARTING POINT
# ==============================================================================
"""
This section demonstrates gradient descent from a VERY BAD starting point:
  m = 600 (way too high — true value is ~28)
  b = -520 (way too low — true value is ~-2)

This tests whether GD can still find the right parameters.
It also generates animations showing:
  1. The regression line fitting the data over epochs
  2. The cost decreasing over epochs
  3. The intercept converging over epochs
  4. The slope converging over epochs
"""

X, y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20, random_state=13)
plt.scatter(X, y)
plt.title("Data Scatter Plot")
plt.show()

# Starting very far from the true solution to demonstrate GD recovery
b = -520      # True intercept ≈ -2.3, starting 518 units away!
m = 600       # True slope ≈ 28, starting 572 units away!
lr = 0.001
all_b, all_m, all_cost = [], [], []
epochs = 30

for i in range(epochs):
    slope_b = 0
    slope_m = 0
    cost = 0
    for j in range(X.shape[0]):
        slope_b = slope_b - 2 * (y[j] - (m * X[j]) - b)
        slope_m = slope_m - 2 * (y[j] - (m * X[j]) - b) * X[j]
        cost = cost + (y[j] - m * X[j] - b) ** 2
    b = b - (lr * slope_b)
    m = m - (lr * slope_m)
    all_b.append(b)
    all_m.append(m)
    all_cost.append(cost)

# ---------------------------------------------------------------------------
# ANIMATION 1: Regression Line Fitting the Data Over Epochs
# ---------------------------------------------------------------------------
"""
This animation shows the regression line moving from its bad starting position
(steep slope, wrong intercept) to eventually fit the data well.
Each frame = one gradient descent update applied.
"""
fig, ax = plt.subplots(figsize=(9, 5))
x_i = np.arange(-3, 3, 0.1)           # X range for drawing the line
ax.scatter(X, y)                        # Static scatter plot of data

line, = ax.plot(x_i, x_i * 50 - 4, 'r-', linewidth=2)   # Initial (wrong) line

def update_line(i):
    """Update regression line for epoch i using stored m and b values."""
    label = f'epoch {i + 1}'
    line.set_ydata(x_i * all_m[i] + all_b[i])  # y = m*x + b with current m,b
    ax.set_xlabel(label)

anim = FuncAnimation(fig, update_line, repeat=True, frames=epochs, interval=500)
# anim.save("animation4.gif", writer=animation.PillowWriter(fps=2))

# ---------------------------------------------------------------------------
# ANIMATION 2: Cost Function Decreasing Over Epochs
# ---------------------------------------------------------------------------
"""
This animation plots COST vs EPOCH NUMBER.
You'll see a DECREASING CURVE — this is what convergence looks like!
The steepest drop is early (big improvements), then it levels off (near optimal).

This is called the "Learning Curve" and it tells you:
  - If curve drops steadily → GD is working well
  - If curve doesn't decrease → learning rate may be too small
  - If curve goes up → learning rate may be too large (overshooting)
"""
num_epochs = list(range(0, 30))

fig = plt.figure(figsize=(9, 5))
axis = plt.axes(xlim=(0, 31), ylim=(0, 4500000))
axis.set_title("Cost vs. Epoch (Learning Curve)")
axis.set_ylabel("Total Cost (Sum of Squared Errors)")
line, = axis.plot([], [], lw=2)
xdata, ydata = [], []

def animate_cost(i):
    """Add one more epoch's cost to the learning curve."""
    xdata.append(num_epochs[i])      # x = epoch number
    ydata.append(all_cost[i])        # y = cost at that epoch
    line.set_data(xdata, ydata)
    axis.set_xlabel(f'epoch {i + 1}')
    return line,

anim = animation.FuncAnimation(fig, animate_cost, frames=30, repeat=False, interval=500)
# anim.save("animation5.gif", writer=animation.PillowWriter(fps=2))

# ---------------------------------------------------------------------------
# ANIMATION 3: Intercept (b) Converging Over Epochs
# ---------------------------------------------------------------------------
"""
Shows how the intercept b starts at -520 and converges toward the true value.
The rapid initial change slows down as we approach the optimum — classic
gradient descent convergence behavior.
"""
fig = plt.figure(figsize=(9, 5))
axis = plt.axes(xlim=(0, 31), ylim=(-10, 160))
axis.set_title("Intercept (b) Convergence Over Epochs")
axis.set_ylabel("b (intercept value)")
line, = axis.plot([], [], lw=2)
xdata, ydata = [], []

def animate_b(i):
    """Plot intercept value at each epoch."""
    xdata.append(num_epochs[i])
    ydata.append(all_b[i])
    line.set_data(xdata, ydata)
    axis.set_xlabel(f'epoch {i + 1}')
    return line,

anim = animation.FuncAnimation(fig, animate_b, frames=30, repeat=False, interval=500)
# anim.save("animation6.gif", writer=animation.PillowWriter(fps=2))

# ---------------------------------------------------------------------------
# ANIMATION 4: Slope (m) Converging Over Epochs
# ---------------------------------------------------------------------------
"""
Shows how the slope m starts at 600 and converges toward ~28.
Similar pattern: rapid initial correction, slower fine-tuning later.
"""
fig = plt.figure(figsize=(9, 5))
axis = plt.axes(xlim=(0, 31), ylim=(-150, 50))
axis.set_title("Slope (m) Convergence Over Epochs")
axis.set_ylabel("m (slope value)")
line, = axis.plot([], [], lw=2)
xdata, ydata = [], []

def animate_m(i):
    """Plot slope value at each epoch."""
    xdata.append(num_epochs[i])
    ydata.append(all_m[i])
    line.set_data(xdata, ydata)
    axis.set_xlabel(f'epoch {i + 1}')
    return line,

anim = animation.FuncAnimation(fig, animate_m, frames=30, repeat=False, interval=500)
# anim.save("animation7.gif", writer=animation.PillowWriter(fps=2))


# ==============================================================================
# SECTION 9: COMPARE SKLEARN LINEAR REGRESSION vs CUSTOM GRADIENT DESCENT
# ==============================================================================
"""
Now we compare two approaches to fit the same data:
  1. sklearn's LinearRegression — uses Ordinary Least Squares (OLS), an
     analytical/mathematical closed-form solution. Exact and fast.
  2. Our custom GDRegressor — uses iterative gradient descent. Approximate
     but more generalizable to complex models (neural networks, etc.)

Both should give very similar results (same dataset, same goal).
The R² score tells us what fraction of variance the model explains:
  R² = 1.0 → perfect fit
  R² = 0.0 → model predicts the mean (useless)
  R² < 0   → model is worse than predicting the mean
"""

X, y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20, random_state=13)

# Split data: 80% train, 20% test
# test_size=0.2 → 20 samples for testing, 80 for training
# random_state=2 → reproducible split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# --- sklearn LinearRegression (analytical OLS) ---
lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train, y_train)

print("=== sklearn LinearRegression (OLS) ===")
print(f"  Slope (m):     {lr_sklearn.coef_[0]:.4f}")
print(f"  Intercept (b): {lr_sklearn.intercept_:.4f}")

y_pred_sklearn = lr_sklearn.predict(X_test)
print(f"  R² Score:      {r2_score(y_test, y_pred_sklearn):.4f}")


# ==============================================================================
# SECTION 10: CUSTOM GRADIENT DESCENT REGRESSOR CLASS
# ==============================================================================
"""
This class packages gradient descent into a reusable, sklearn-style API.
It mimics sklearn's interface: .fit(X, y) to train, .predict(X) to infer.

DESIGN NOTES:
  - Uses VECTORIZED operations (np.sum) instead of inner loop → faster
  - np.sum(a * b) is equivalent to Σ(a_i * b_i) — element-wise multiply then sum
  - X.ravel() flattens 2D array to 1D for element-wise operations with 1D y
"""

class GDRegressor:
    """
    Linear Regression trained with Batch Gradient Descent.
    
    Math:
      Model: y_pred = m*x + b
      Cost:  L(m,b) = Σ(y - m*x - b)²
      
    Gradient Update:
      ∂L/∂b = -2 * Σ(y - m*x - b)
      ∂L/∂m = -2 * Σ(y - m*x - b) * x
      
      b_new = b - lr * ∂L/∂b
      m_new = m - lr * ∂L/∂m
    """

    def __init__(self, learning_rate, epochs):
        """
        Initialize with arbitrary starting parameters.
        
        Args:
          learning_rate (float): Step size for each gradient update.
                                 Typical values: 0.001, 0.01, 0.1
          epochs (int):          Number of training iterations over full dataset.
        """
        self.m = 100          # Initial slope — arbitrary starting point
        self.b = -120         # Initial intercept — arbitrary starting point
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Args:
          X: Feature array, shape (n_samples, 1)
          y: Target array, shape (n_samples,)
          
        After training, self.m and self.b hold the learned parameters.
        """
        for i in range(self.epochs):
            # VECTORIZED gradient computation (no inner loop needed!)
            # np.sum computes the full sum in one shot
            
            # Residuals: how far off our predictions are
            residuals = y - self.m * X.ravel() - self.b  # shape: (n_samples,)
            
            # Gradient for b: ∂L/∂b = -2 * Σ residuals
            loss_slope_b = -2 * np.sum(residuals)
            
            # Gradient for m: ∂L/∂m = -2 * Σ (residuals * x)
            # X.ravel() × residuals = element-wise multiplication
            loss_slope_m = -2 * np.sum(residuals * X.ravel())

            # Update parameters — move opposite to gradient direction
            self.b = self.b - (self.lr * loss_slope_b)
            self.m = self.m - (self.lr * loss_slope_m)

        print(f"Learned slope (m):     {self.m:.4f}")
        print(f"Learned intercept (b): {self.b:.4f}")

    def predict(self, X):
        """
        Predict target values for input X using learned parameters.
        
        Args:
          X: Feature array, shape (n_samples, 1)
          
        Returns:
          y_pred: Predicted values, shape (n_samples, 1)
        """
        return self.m * X + self.b   # y = mx + b for all rows of X at once


# --- Train and evaluate our custom GDRegressor ---
gd = GDRegressor(learning_rate=0.001, epochs=50)
gd.fit(X_train, y_train)

y_pred_gd = gd.predict(X_test)

print("\n=== Custom GDRegressor (Gradient Descent) ===")
print(f"  R² Score: {r2_score(y_test, y_pred_gd):.4f}")
print("\nNote: Both R² scores should be very close — GD finds nearly the same solution as OLS!")


# ==============================================================================
# SECTION 11: STEP-BY-STEP GRADIENT DESCENT (Small Dataset, Manual Walkthrough)
# ==============================================================================
"""
This section uses only 4 data points to manually trace through gradient descent
step by step, making the math extremely concrete and easy to follow.

We fix m = 78.35 (true slope) and only optimize b (intercept).
This simplifies the problem to 1D gradient descent.
"""

X, y = make_regression(n_samples=4, n_features=1, n_informative=1, n_targets=1, noise=80, random_state=13)
plt.scatter(X, y, s=100)
plt.title("4-Point Dataset for Manual GD Walkthrough")
plt.show()

# Fit sklearn to see the "correct" answer
reg = LinearRegression()
reg.fit(X, y)
print(f"OLS solution: m={reg.coef_[0]:.4f}, b={reg.intercept_:.4f}")

# Fix slope at true value, start intercept at 100 (true is ~26)
m = 78.35   # Known slope (fixing this so we only tune b)
b = 100     # Starting intercept (way off!)
lr = 0.1    # Learning rate (larger for only 4 points — fewer samples → can use bigger LR)

# ---- ITERATION 1 ----
"""
Loss slope formula: -2 * Σ(y_actual - y_predicted)
                  = -2 * Σ(y - m*x - b)
"""
loss_slope = -2 * np.sum(y - m * X.ravel() - b)
print(f"\nIteration 1:")
print(f"  loss_slope = {loss_slope:.4f}")   # Positive → cost increasing as b increases → decrease b

step_size = loss_slope * lr   # How much to move
print(f"  step_size  = {step_size:.4f}")

b = b - step_size             # Update: move opposite to gradient
print(f"  new b      = {b:.4f}")

y_pred = ((78.35 * X) + b).reshape(4)

# ---- ITERATION 2 ----
loss_slope = -2 * np.sum(y - m * X.ravel() - b)
print(f"\nIteration 2:")
print(f"  loss_slope = {loss_slope:.4f}")   # Smaller than before → getting closer to min

step_size = loss_slope * lr
print(f"  step_size  = {step_size:.4f}")   # Smaller step — gradient is smaller near minimum

b = b - step_size
print(f"  new b      = {b:.4f}")   # Getting closer to ~26

# ---- ITERATION 3 ----
loss_slope = -2 * np.sum(y - m * X.ravel() - b)
print(f"\nIteration 3:")
print(f"  loss_slope = {loss_slope:.4f}")   # Even smaller now
step_size = loss_slope * lr
print(f"  step_size  = {step_size:.4f}")
b = b - step_size
print(f"  new b      = {b:.4f}")   # Very close to OLS answer of 26.16 now!

"""
KEY OBSERVATION: Each iteration, the gradient (loss_slope) gets smaller and the
step_size shrinks. This is why gradient descent naturally slows down as it
approaches the minimum — the terrain flattens out near the bottom of the bowl!
"""


# ==============================================================================
# SECTION 12: FULL LOOP GRADIENT DESCENT (100 Epochs, Visualized)
# ==============================================================================
"""
Now we run full gradient descent for 100 epochs, plotting the regression line
at every epoch. You'll see the line "chase" the best fit from a bad start.
"""

b = -100   # Start far below true value
m = 78.35  # Keep slope fixed for this demonstration
lr = 0.01  # Slightly higher learning rate for faster convergence

for i in range(100):
    # Vectorized gradient: equivalent to the inner j-loop but faster
    loss_slope = -2 * np.sum(y - m * X.ravel() - b)
    b = b - (lr * loss_slope)

    # Plot regression line at this epoch
    y_pred_line = m * X + b
    plt.plot(X, y_pred_line, alpha=0.3, color='blue')  # alpha makes lines semi-transparent

plt.scatter(X, y, color='red', zorder=5, s=100)   # Data points on top
plt.title("All 100 Epochs of Gradient Descent\n(Lines converging toward best fit)")
plt.xlabel("X")
plt.ylabel("y")
plt.show()


# ==============================================================================
# SECTION 13: ONLY-b GRADIENT DESCENT WITH ANIMATIONS (Fixed m Version)
# ==============================================================================
"""
This section demonstrates gradient descent on a larger dataset (100 points),
but fixing m and only learning b. Great for understanding the b-convergence
in isolation.

Key variable: m = 27.82 (close to true value), b starts at -150.
"""

X, y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20, random_state=13)

reg = LinearRegression()
reg.fit(X, y)
print(f"\nFull dataset OLS: m={reg.coef_[0]:.4f}, b={reg.intercept_:.4f}")

b = -150    # Start far below true value (-2.29)
m = 27.82   # Close to true slope (27.83)
lr = 0.001  # Conservative learning rate for 100 data points
all_b, all_cost = [], []
epochs = 30

for i in range(epochs):
    slope = 0
    cost = 0
    for j in range(X.shape[0]):
        # Only computing slope for b (not m, since m is fixed here)
        slope = slope - 2 * (y[j] - (m * X[j]) - b)
        cost = cost + (y[j] - m * X[j] - b) ** 2

    b = b - (lr * slope)
    all_b.append(b)
    all_cost.append(cost)

    # Plot regression line at each epoch
    y_pred = m * X + b
    plt.plot(X, y_pred, alpha=0.3, color='blue')

plt.scatter(X, y, color='red', zorder=5)
plt.title("Gradient Descent (only b): Line Fitting Progress")
plt.show()

# Show convergence arrays
all_b = np.array(all_b).ravel()
all_cost = np.array(all_cost).ravel()

print(f"\nIntercept values over {epochs} epochs:")
print(all_b)
print(f"\nTrue intercept: {reg.intercept_:.4f}")
print(f"GD intercept after {epochs} epochs: {all_b[-1]:.4f}")
print(f"\nCost values over {epochs} epochs:")
print(all_cost)
print(f"\nStarting cost: {all_cost[0]:,.0f}")
print(f"Final cost:    {all_cost[-1]:,.0f}")
print(f"Cost reduction: {(1 - all_cost[-1]/all_cost[0])*100:.1f}%")


# ==============================================================================
# SUMMARY: KEY TAKEAWAYS
# ==============================================================================
"""
SUMMARY OF WHAT WE LEARNED:
══════════════════════════════

1. COST FUNCTION: Measures how wrong our model is. Lower = better.
   Formula: Σ(y_actual - y_predicted)²

2. GRADIENT: The slope of the cost surface at our current (m, b) position.
   Points in the direction of steepest increase.

3. GRADIENT DESCENT: Always moves OPPOSITE to the gradient (downhill).
   Update rule: param = param - learning_rate × gradient

4. LEARNING RATE:
   - Too high → oscillates, may diverge (overshoot the valley)
   - Too low  → converges, but very slowly
   - Just right → converges steadily and quickly

5. EPOCHS: More iterations → closer to true minimum, but returns diminish.

6. CONVERGENCE: Gradient descent converges when gradients become ~0 (flat ground).

7. BATCH vs STOCHASTIC:
   - Batch GD (used here): Use ALL data per update → stable, but slow per iteration
   - Stochastic GD: Use ONE sample per update → noisy, but faster
   - Mini-batch GD: Use a small batch → balance of both (used in deep learning)

8. GRADIENT DESCENT vs OLS:
   - OLS: Exact, fast, only for linear regression
   - GD:  Approximate, iterative, works for any model (neural nets, etc.)
"""


# ==============================================================================
# SECTION 14: cost_function.html — 3D COST SURFACE (INTERACTIVE)
# ==============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              WHAT IS cost_function.html?                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

This file is a self-contained interactive 3D visualization generated by Plotly.
Open it in any browser — no Python/Jupyter needed.

WHAT YOU SEE:
─────────────
  A beautiful 3D BOWL-SHAPED SURFACE where:
    • X-axis  → m (slope values from -150 to +150)
    • Y-axis  → b (intercept values from -150 to +150)
    • Z-axis  → Cost = Σ(y - mx - b)²  at each (m, b) pair
    • Color   → Same as Z (low cost = dark/cool, high cost = bright/warm)

  The LOWEST POINT of the bowl is the optimal (m*, b*) — the best fit line.
  Every other point on the surface means the model is MORE wrong.

WHY IS IT BOWL-SHAPED?
───────────────────────
  The cost function is a CONVEX QUADRATIC in both m and b.
  Squaring the errors guarantees:
    1. No negative values (all costs ≥ 0)
    2. Exactly ONE global minimum (no local minima to get stuck in)
    3. Smooth, bowl-like surface (differentiable everywhere)

  This is why gradient descent is GUARANTEED to find the global minimum
  for linear regression — the loss landscape has no traps!

INTERACTION TIPS:
──────────────────
  • Left-click + drag → rotate the 3D bowl
  • Scroll             → zoom in/out
  • Right-click + drag → pan
  • Hover over surface → see exact (m, b, cost) values at any point
  • Click "Download plot" (camera icon) → save as PNG

HOW IT WAS GENERATED:
──────────────────────
  Step 1: Create grid of 10×10 = 100 (m, b) combinations
          m_arr = np.linspace(-150, 150, 10)   → [-150, -117, -83, ..., 150]
          b_arr = np.linspace(-150, 150, 10)   → same range
          mGrid, bGrid = np.meshgrid(m_arr, b_arr)  → 2D arrays

  Step 2: For each (m, b) pair, compute cost:
          cost = Σ (y_actual - m*x - b)²  over all 100 training points

  Step 3: Reshape 100 costs → 10×10 matrix (z_arr)

  Step 4: Plot with plotly go.Surface(x=m_arr, y=b_arr, z=z_arr)

  Step 5: fig.write_html("cost_function.html") → saves standalone HTML
          (Plotly embeds all its JavaScript inside the file — it's self-contained!)

MATHEMATICAL INSIGHT:
──────────────────────
  The minimum of this surface occurs at:
    m* = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²     ← OLS formula for slope
    b* = ȳ - m* × x̄                         ← OLS formula for intercept

  Gradient descent finds this same minimum, but iteratively instead of
  solving it analytically. Both methods reach the same (m*, b*)!
"""

# Code that generates cost_function.html:
from sklearn.datasets import make_regression
import numpy as np
import plotly.graph_objects as go

X, y = make_regression(n_samples=100, n_features=1, n_informative=1,
                       n_targets=1, noise=20, random_state=13)

# ── STEP 1: Build the parameter grid ──────────────────────────────────────────
# We want to EVALUATE the cost at many different (m, b) combinations.
# linspace creates 10 evenly spaced values in [-150, 150]
m_arr = np.linspace(-150, 150, 10)   # 10 candidate slopes
b_arr = np.linspace(-150, 150, 10)   # 10 candidate intercepts

# meshgrid creates a 2D grid of all combinations
# mGrid[i,j] = m_arr[j],  bGrid[i,j] = b_arr[i]
# → Total combinations: 10 × 10 = 100 (m, b) pairs
mGrid, bGrid = np.meshgrid(m_arr, b_arr)

# Flatten and stack into (100, 2) array — each row = one (m, b) pair to test
final = np.vstack((
    mGrid.ravel().reshape(1, 100),   # row 0: all m values
    bGrid.ravel().reshape(1, 100)    # row 1: all b values
)).T                                  # Transpose → (100, 2)

# ── STEP 2: Compute cost at each (m, b) ───────────────────────────────────────
z_arr = []
for i in range(final.shape[0]):       # loop over all 100 (m, b) pairs
    m_val = final[i, 0]               # extract this pair's slope
    b_val = final[i, 1]               # extract this pair's intercept
    # Cost = Sum of Squared Residuals across all 100 training points
    # residual = (actual y) - (predicted y) = y - (m*x + b)
    cost = np.sum((y - m_val * X.reshape(100) - b_val) ** 2)
    z_arr.append(cost)

# Reshape 100 costs into 10×10 matrix (rows=b values, cols=m values)
z_arr = np.array(z_arr).reshape(10, 10)

# ── STEP 3: Plot the 3D surface ───────────────────────────────────────────────
fig = go.Figure(data=[go.Surface(
    x=m_arr,            # X-axis: slope values
    y=b_arr,            # Y-axis: intercept values
    z=z_arr,            # Z-axis: cost at each (m, b)
    colorscale='Viridis'  # Color scheme: purple (low) → yellow (high)
)])

fig.update_layout(
    title='Cost Function Surface — Find the Bowl\'s Minimum!',
    autosize=False,
    width=800, height=600,
    scene=dict(
        xaxis_title='m (slope)',
        yaxis_title='b (intercept)',
        zaxis_title='Cost = Σ(y - mx - b)²'
    ),
    margin=dict(l=65, r=50, b=65, t=90)
)

fig.show()
fig.write_html("cost_function.html")   # ← This creates the standalone HTML file
# The HTML is fully self-contained: Plotly JS is embedded inside.
# Share this file with anyone — they just open it in a browser!


# ==============================================================================
# SECTION 15: cost_function2.html — GD PATH TRACED ON 3D SURFACE (INTERACTIVE)
# ==============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              WHAT IS cost_function2.html?                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

This is the MOST IMPORTANT visualization — it combines:
  1. The same 3D cost function bowl (blue surface, semi-transparent)
  2. The gradient descent PATH as colored dots on the surface

It answers the question: "Where exactly does gradient descent walk on the bowl?"

WHAT YOU SEE:
─────────────
  • BLUE TRANSPARENT SURFACE → The cost function bowl (same as cost_function.html)
  • COLORED DOTS (scatter3d) → Each dot = one epoch of gradient descent
    - Dot position = (m_at_epoch, b_at_epoch, cost_at_epoch × 100)
    - Color changes with cost (bright = expensive, dark = cheap)
    - The dots form a PATH from high cost → low cost

  Starting point: m=-127.82, b=150, cost=very high (up the hill)
  Final point:    m≈28,      b≈-2,  cost=minimum (bottom of bowl)

WHY MULTIPLY COST BY 100?
───────────────────────────
  `z=np.array(all_cost).ravel() * 100`
  
  Pure visual trick! The GD path dots would float UNDER the surface without
  scaling, making them invisible. Multiplying by 100 lifts them to sit ON
  the surface, making the path clearly visible against the bowl.
  The actual math is unchanged — only the display scale is adjusted.

THE PATH SHAPE — WHAT TO NOTICE:
──────────────────────────────────
  1. STEEP DROP early → large gradients early on = big steps = fast initial progress
  2. FLATTENING later → small gradients near minimum = tiny steps = fine-tuning
  3. PATH CURVES → because m and b are coupled — updating one changes the optimal other
  4. SPIRALING → often seen when learning rate is slightly large (oscillations)
  5. CONVERGENCE → path stops moving when gradient ≈ 0 (flat ground)

INTERACTION TIPS:
──────────────────
  • Rotate to see the path "descending" into the bowl from above
  • Look at the path from ABOVE (top view) — it traces the contour map
  • Hover over individual dots → see exact epoch, m, b, cost values
  • Toggle surface on/off by clicking its legend entry
  • Zoom into the endpoint to see where the path converges

WHAT CHANGES IF YOU CHANGE HYPERPARAMETERS:
────────────────────────────────────────────
  LEARNING RATE too HIGH (e.g., lr=0.1):
    → Path jumps wildly back and forth (overshoots the valley)
    → May diverge (cost goes UP instead of down)
    → Dots scattered chaotically, not following a smooth descent

  LEARNING RATE too LOW (e.g., lr=0.00001):
    → Path moves extremely slowly
    → Need many more epochs to reach the minimum
    → Dots clustered near starting point after 30 epochs

  MORE EPOCHS (e.g., 200):
    → Path continues further, reaching closer to the true minimum
    → Later dots converge tightly at the bowl's base

  DIFFERENT STARTING POINT:
    → Path starts from a different location on the bowl
    → Always converges to the SAME minimum (convex function!)
    → This is why linear regression GD is robust to initialization

HOW IT WAS GENERATED:
──────────────────────
  Step 1: Run 30 epochs of gradient descent, recording (m, b, cost) each epoch
  Step 2: Plot the (m, b, cost×100) values as a 3D scatter plot (the path dots)
  Step 3: Add the cost surface as a semi-transparent background
  Step 4: Combine both traces in one Plotly figure → write_html
"""

import plotly.express as px

# ── STEP 1: Run Gradient Descent and record the path ──────────────────────────
X, y = make_regression(n_samples=100, n_features=1, n_informative=1,
                       n_targets=1, noise=20, random_state=13)

b = 150       # Intentionally bad starting intercept (true value ≈ -2)
m = -127.82   # Intentionally bad starting slope (true value ≈ +28)
lr = 0.001    # Learning rate: small enough to not overshoot

all_b, all_m, all_cost = [], [], []   # History lists — record every epoch

epochs = 30

for i in range(epochs):
    slope_b = 0   # Will accumulate ∂Cost/∂b  across all data points
    slope_m = 0   # Will accumulate ∂Cost/∂m  across all data points
    cost    = 0   # Will accumulate total cost across all data points

    for j in range(X.shape[0]):   # Loop over all 100 training examples
        # Error for this sample: how wrong is our current prediction?
        error = y[j] - (m * X[j]) - b   # residual = actual - predicted

        # Partial derivatives (∂Cost/∂b and ∂Cost/∂m) for this sample:
        # We negate because: slope_b accumulates with a minus sign,
        # making slope_b = -2 * Σ(error) which IS the gradient ∂Cost/∂b
        slope_b = slope_b - 2 * error             # accumulate gradient for b
        slope_m = slope_m - 2 * error * X[j]      # accumulate gradient for m
        cost    = cost    + error ** 2             # accumulate squared error

    # Parameter update — the core of gradient descent:
    # Move each parameter OPPOSITE to its gradient direction
    b = b - (lr * slope_b)   # new intercept (closer to optimal)
    m = m - (lr * slope_m)   # new slope (closer to optimal)

    # Record this epoch's values for the 3D path visualization
    all_b.append(b)        # WHERE we are in b-space
    all_m.append(m)        # WHERE we are in m-space
    all_cost.append(cost)  # HOW WRONG we still are

# ── STEP 2: Create the GD path as colored 3D scatter dots ─────────────────────
# Each dot = one epoch. Color encodes cost (bright = expensive = far from minimum)
fig = px.scatter_3d(
    x=np.array(all_m).ravel(),         # X position = slope at this epoch
    y=np.array(all_b).ravel(),         # Y position = intercept at this epoch
    z=np.array(all_cost).ravel() * 100,# Z position = cost (×100 for visibility)
    labels={
        'x': 'm (slope)',
        'y': 'b (intercept)',
        'z': 'Cost × 100'
    },
    title='Gradient Descent Path Descending the Cost Function Bowl'
    # Color automatically maps to cost value — bright = high cost, dark = low
)

# ── STEP 3: Add the cost function surface as the background bowl ───────────────
# We reuse m_arr, b_arr, z_arr computed in Section 14
# opacity=0.7 makes it semi-transparent so the GD path dots are visible through it
fig.add_trace(go.Surface(
    x=m_arr,                   # Same slope axis as the path
    y=b_arr,                   # Same intercept axis as the path
    z=z_arr * 100,             # ×100 to match the path's Z scale
    colorscale='Blues',        # Cool blue color scheme for the surface
    opacity=0.7,               # Semi-transparent → can see path through surface
    showscale=False            # Hide surface's own color legend (path has its own)
))

fig.show()
fig.write_html("cost_function2.html")  # Save as standalone interactive HTML

# ── READING THE OUTPUT ────────────────────────────────────────────────────────
print("Gradient Descent Results (30 epochs):")
print(f"  Starting: m={-127.82:.2f}, b={150:.2f}")
print(f"  Final:    m={all_m[-1]:.4f}, b={all_b[-1]:.4f}")
print(f"  Starting cost: {all_cost[0]:>15,.0f}")
print(f"  Final cost:    {all_cost[-1]:>15,.0f}")
print(f"  Cost reduced by: {(1 - all_cost[-1]/all_cost[0])*100:.2f}%")
print(f"\n  True sklearn solution: m≈28, b≈-2")
print(f"  Our GD solution:       m≈{all_m[-1]:.1f}, b≈{all_b[-1]:.1f}")
print(f"  ✓ Very close! More epochs → even closer.")


# ==============================================================================
# SECTION 16: BOTH HTML FILES — SIDE-BY-SIDE COMPARISON
# ==============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          cost_function.html  vs  cost_function2.html                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  cost_function.html           │  cost_function2.html                        ║
║  ─────────────────────────── │ ──────────────────────────────────           ║
║  Shows: Cost LANDSCAPE only   │  Shows: Landscape + GD PATH                 ║
║  Purpose: "What does the      │  Purpose: "Where does GD walk               ║
║   bowl look like?"            │   on this bowl?"                            ║
║  Traces: 1 (Surface)          │  Traces: 2 (Surface + Scatter3D)            ║
║  Best for: Understanding WHY  │  Best for: Understanding HOW                ║
║   we minimize cost            │   gradient descent optimizes                ║
║  Analogy: Seeing the mountain │  Analogy: Watching the hiker                ║
║   from a helicopter           │   descend the mountain                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

TOGETHER, THEY TELL THE COMPLETE STORY:
  1. cost_function.html  → "Here is the terrain" (the optimization landscape)
  2. cost_function2.html → "Here is the journey" (gradient descent navigating it)

This is one of the most beautiful visualizations in all of introductory ML:
watching an algorithm literally "roll downhill" to find the answer.

WHAT THE OUTPUTS PROVE:
  • The cost function bowl has a single, clear minimum → linear regression is convex
  • Gradient descent finds it reliably, regardless of starting position
  • The path isn't perfectly straight (m and b interact), but always converges
  • After 30 epochs with lr=0.001, we get m≈28, b≈-2 ← same as sklearn's OLS!
"""
