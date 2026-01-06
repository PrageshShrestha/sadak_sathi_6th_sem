import pandas as pd
import numpy as np
df = pd.read_csv("lake_updated.csv")
df.columns = ["from","to","dist" , "time" , "dir_ref" , "rush"]
# Separate features (X) and target variable (y)
X = df[['from', 'to' ,"dist",  "dir_ref" , "rush"]]
y = df['time']
df.head(20)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test) # If you have a test set
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',             # We track RMSE
    'eta': 0.1,                        # Moderate learning rate (0.05–0.3 works well for overfitting)
    'max_depth': 0,                    # 0 = no limit → grow extremely deep trees (key for overfitting!)
    'min_child_weight': 1,             # Minimum = 1 → almost no restriction on leaf nodes
    'gamma': 0.0,                      # No minimum loss reduction → allow all splits
    'subsample': 1.0,                  # Use 100% of rows → no row bagging
    'colsample_bytree': 1.0,           # Use 100% of features per tree → no column bagging
    'colsample_bylevel': 1.0,          # Use all features at each level
    'colsample_bynode': 1.0,           # Use all features at each node
    'reg_lambda': 0.0,                 # No L2 regularization
    'reg_alpha': 0.0,                  # No L1 regularization
    'seed': 27,
    'tree_method': 'hist',          # Faster training with deep trees
    'device': 'cuda'                # Uncomment if you have GPU (much faster for deep trees)
}
num_rounds = 100  #
import xgboost as xgb
from xgboost.callback import TrainingCallback
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

print("Starting XGBoost regression training...")

# ==================== REGRESSION PARAMETERS (NO scale_pos_weight!) ====================


# ==================== SAFE & ROBUST CALLBACK ====================
class LoggingCallback(TrainingCallback):
    def __init__(self):
        super().__init__()
        self.metrics = []
        self.epoch_times = []
        self.epoch_start = time.time()

    def after_iteration(self, model, epoch, evals_log):
        # Measure time for this iteration
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)

        # Safety: skip if no eval log yet (rare but possible on iter 0)
        if not evals_log:
            print(f"[{epoch+1:4d}] No metrics yet | time: {epoch_time:.2f}s")
            self.epoch_start = time.time()
            return False

        # Get the metric name safely (should be 'rmse')
        metric_name = next(iter(evals_log['train']))  # First and only key

        # Get values safely
        train_val = evals_log['train'][metric_name][-1]
        test_val = evals_log['test'][metric_name][-1] if 'test' in evals_log else None

        # Print progress
        print_str = f"[{epoch+1:4d}] train-{metric_name}: {train_val:.6f} | time: {epoch_time:.2f}s"
        if test_val is not None:
            print_str += f" | test-{metric_name}: {test_val:.6f}"

        # Record metrics
        self.metrics.append({
            'iteration': epoch + 1,
            'dataset': 'train',
            'metric': metric_name,
            'value': train_val
        })
        if test_val is not None:
            self.metrics.append({
                'iteration': epoch + 1,
                'dataset': 'test',
                'metric': metric_name,
                'value': test_val
            })

        # Reset timer for next iteration
        self.epoch_start = time.time()
        return False

# ==================== TRAINING WITH CALLBACK ====================
log_callback = LoggingCallback()

start_total = time.time()

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_rounds,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    callbacks=[log_callback],       # ← NOW ACTUALLY USED
    verbose_eval=False              # ← Let callback handle printing
)
model.save_model("model_eval/xgb_model.pkl")
total_time = time.time() - start_total
print(f"\nTraining finished in {total_time:.2f}s ({total_time/60:.2f} min)")

# ==================== SAFE PLOTTING ====================
if not log_callback.metrics:
    print("ERROR: No metrics were collected. Training may have failed.")
else:
    metrics_df = pd.DataFrame(log_callback.metrics)
    
    # Simple and safe pivot: separate columns for train_rmse and test_rmse
    plot_df = metrics_df.pivot(index='iteration', columns='dataset', values='value').reset_index()
    plot_df.columns = ['iteration', 'train_rmse', 'test_rmse']  # Clear column names
    
    # Cumulative time (aligned to end of each iteration)
    plot_df['cumulative_time'] = pd.Series(log_callback.epoch_times).cumsum()

    # Plot 1: RMSE over iterations
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df, x='iteration', y='train_rmse', label='Training RMSE', linewidth=2.5)
    if 'test_rmse' in plot_df.columns and not plot_df['test_rmse'].isna().all():
        sns.lineplot(data=plot_df, x='iteration', y='test_rmse', label='Validation RMSE', linewidth=2.5)
    plt.title('XGBoost Regression: RMSE over Iterations', fontsize=16)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: RMSE vs Time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df, x='cumulative_time', y='train_rmse', label='Training RMSE')
    if 'test_rmse' in plot_df.columns and not plot_df['test_rmse'].isna().all():
        sns.lineplot(data=plot_df, x='cumulative_time', y='test_rmse', label='Validation RMSE')
    
    plt.title('RMSE vs Training Time')
    plt.xlabel('Cumulative Time (seconds)')
    plt.ylabel('RMSE')
    plt.legend()
    plt.tight_layout()
    plt.show()
    import matplotlib.pyplot as plt
import numpy as np

# Assuming you have:
predictions = model.predict(dtest)

# y_test = actual test targets (as pandas Series or numpy array)

# Convert to numpy for easier indexing
actual = y_test.values if hasattr(y_test, 'values') else y_test
pred = predictions

# Use first N samples for clear visualization (e.g., 50–100 to avoid clutter)
N = 80
indices = np.arange(N)

plt.figure(figsize=(14, 7))

# Width of bars
width = 0.35

# Side-by-side bar plot: Actual vs Predicted
plt.bar(indices - width/2, actual[:N], width, label='Actual Time', color='skyblue', alpha=0.8)
plt.bar(indices + width/2, pred[:N], width, label='Predicted Time', color='salmon', alpha=0.8)

plt.xlabel('Sample Index (Test Set)', fontsize=12)
plt.ylabel('Travel Time', fontsize=12)
plt.title('Actual vs Predicted Travel Time (First 80 Test Samples)', fontsize=16)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

# Optional: Add a line connecting actual and predicted for each point (helps see errors)
for i in range(N):
    plt.plot([indices[i] - width/2, indices[i] + width/2], [actual[i], pred[i]], 
             color='gray', linewidth=0.8, alpha=0.6)

plt.tight_layout()
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Prepare data
actual = y_test.values if hasattr(y_test, 'values') else y_test
pred = predictions

# Calculate metrics
r2 = r2_score(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(10, 8))

# Seaborn scatter plot
sns.scatterplot(x=actual, y=pred, alpha=0.6, color='teal', s=60)  # s controls point size

# Add the perfect prediction line (y = x)
min_val = min(actual.min(), pred.min())
max_val = max(actual.max(), pred.max())

# Use Seaborn's lineplot for the diagonal reference line
sns.lineplot(x=[min_val, max_val], y=[min_val, max_val], 
             color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')

# Labels and title
plt.xlabel('Actual Travel Time (minutes)', fontsize=12)
plt.ylabel('Predicted Travel Time (minutes)', fontsize=12)
plt.title(f'Predicted vs Actual Travel Time\n(R² = {r2:.3f} | RMSE = {rmse:.2f})', 
          fontsize=14, pad=20)

# Legend (Seaborn places it automatically, but we can tweak if needed)
plt.legend(fontsize=11)

# Final adjustments
plt.tight_layout()
plt.show()
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"\nModel Evaluation on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")
model.save_model("./model_eval/xgb_model_final.pkl")