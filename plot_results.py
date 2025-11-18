import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load results
with open('results_summary.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results).T

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('ML Model Performance Comparison', fontsize=16)

# 1. Accuracy
axes[0,0].bar(df.index, df['accuracy'])
axes[0,0].set_title('Accuracy')
axes[0,0].set_ylabel('Score')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Sharpe Ratio
colors = ['red' if x < 0 else 'green' for x in df['sharpe']]
axes[0,1].bar(df.index, df['sharpe'], color=colors)
axes[0,1].set_title('Sharpe Ratio')
axes[0,1].set_ylabel('Sharpe')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# 3. Total Return
colors = ['red' if x < 0 else 'green' for x in df['total_return']]
axes[0,2].bar(df.index, df['total_return'], color=colors)
axes[0,2].set_title('Total Return')
axes[0,2].set_ylabel('Return')
axes[0,2].tick_params(axis='x', rotation=45)
axes[0,2].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# 4. AUC
axes[1,0].bar(df.index, df['auc'])
axes[1,0].set_title('AUC Score')
axes[1,0].set_ylabel('AUC')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')

# 5. F1 Score
axes[1,1].bar(df.index, df['f1'])
axes[1,1].set_title('F1 Score')
axes[1,1].set_ylabel('F1')
axes[1,1].tick_params(axis='x', rotation=45)

# 6. Annualized Volatility
axes[1,2].bar(df.index, df['ann_vol'])
axes[1,2].set_title('Annualized Volatility')
axes[1,2].set_ylabel('Volatility')
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/results_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary table
print("\nModel Performance Summary:")
print("=" * 80)
print(f"{'Model':<8} {'Accuracy':<8} {'Sharpe':<8} {'Return':<8} {'AUC':<6} {'F1':<6}")
print("-" * 80)
for model in df.index:
    print(f"{model:<8} {df.loc[model, 'accuracy']:<8.3f} {df.loc[model, 'sharpe']:<8.2f} "
          f"{df.loc[model, 'total_return']:<8.2f} {df.loc[model, 'auc']:<6.3f} "
          f"{df.loc[model, 'f1']:<6.3f}")

# Best performers
print(f"\nBest Sharpe Ratio: {df['sharpe'].idxmax()} ({df['sharpe'].max():.2f})")
print(f"Best Total Return: {df['total_return'].idxmax()} ({df['total_return'].max():.2f})")
print(f"Best Accuracy: {df['accuracy'].idxmax()} ({df['accuracy'].max():.3f})")