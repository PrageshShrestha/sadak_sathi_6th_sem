import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import os
from tqdm import tqdm  # For batch progress bar

# --- Load your data ---
df = pd.read_csv("lake_updated.csv", header=None)
df.columns = ['from_node', 'to_node', 'dist', 'time', 'dir_ref', 'rush']

# --- Preprocessing ---
cat_cols = ['from_node', 'to_node', 'dir_ref', 'rush']
num_cols = ['dist']

# Vocabulary sizes for embeddings (add padding if needed)
vocab_sizes = {
    'from_node': int(df['from_node'].max() + 1),
    'to_node':   int(df['to_node'].max() + 1),
    'dir_ref':   int(df['dir_ref'].max() + 1),
    'rush':      int(df['rush'].max() + 1),
}

# Scale numerical feature
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Features and target
X = df[cat_cols + num_cols]
y = df['time']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Dataset ---
class ETADataset(Dataset):
    def __init__(self, X, y):
        self.cat = torch.tensor(X[cat_cols].values, dtype=torch.long)
        self.num = torch.tensor(X[num_cols].values, dtype=torch.float32)
        self.target = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return {
            'cat': self.cat[idx],
            'num': self.num[idx],
            'target': self.target[idx]
        }

train_dataset = ETADataset(X_train, y_train)
test_dataset  = ETADataset(X_test, y_test)

train_loader = DataLoader(train_dataset,num_workers = 4, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_dataset,num_workers = 4, batch_size=256, shuffle=False)

# --- Model (FT-Transformer style) ---
class FTTransformerRegressor(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=32, depth=4, heads=4, ff_hidden=128, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_sizes[col], embed_dim) for col in cat_cols
        ])
        num_numerical = len(num_cols)
        self.num_proj = nn.Linear(num_numerical, embed_dim) if num_numerical > 0 else None
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, cat, num):
        batch_size = cat.shape[0]
        
        embeds = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.stack(embeds, dim=1)  # [batch, n_cat, embed_dim]
        
        if self.num_proj is not None:
            num_emb = self.num_proj(num).unsqueeze(1)
            x = torch.cat([x, num_emb], dim=1)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.transformer(x)
        cls_out = x[:, 0, :]
        
        return self.head(cls_out).squeeze(-1)

# --- Device setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = FTTransformerRegressor(
    vocab_sizes=vocab_sizes,
    embed_dim=32,
    depth=4,
    heads=4,
    ff_hidden=128,
    dropout=0.2
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)

# --- Training with full epoch logging ---
epochs = 30
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 30

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    # tqdm for batch-level progress
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        cat = batch['cat'].to(device)
        num = batch['num'].to(device)
        target = batch['target'].to(device)
        
        optimizer.zero_grad()
        pred = model(cat, num)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            cat = batch['cat'].to(device)
            num = batch['num'].to(device)
            target = batch['target'].to(device)
            pred = model(cat, num)
            val_loss += criterion(pred, target).item()
    
    val_loss /= len(test_loader)
    scheduler.step(val_loss)
    
    # Full logging every epoch
    print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6e}")

    # Early stopping + save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_eta_transformer.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# --- Final Evaluation ---
model.load_state_dict(torch.load('best_eta_transformer.pth'))
model.eval()

predictions = []
actuals = []
with torch.no_grad():
    for batch in test_loader:
        cat = batch['cat'].to(device)
        num = batch['num'].to(device)
        pred = model(cat, num)
        predictions.extend(pred.cpu().numpy())
        actuals.extend(batch['target'].cpu().numpy())

predictions = np.array(predictions)
actuals = np.array(actuals)

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)

print("\n=== Final Model Performance on Test Set ===")
print(f"MAE : {mae:.6f} minutes")
print(f"RMSE: {rmse:.6f} minutes")
print(f"R²  : {r2:.6f}")

# --- Save models ---
os.makedirs('model_eval', exist_ok=True)
torch.save(model, 'model_eval/eta_transformer_full.pth')
torch.save(model.state_dict(), 'model_eval/eta_transformer_state.pth')
print("\nModel saved successfully in './model_eval/'")
