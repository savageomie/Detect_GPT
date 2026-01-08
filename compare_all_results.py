# ============================================================================
# COMPREHENSIVE COMPARISON: All AI Text Detection Approaches
# Compares: Feature-Based ML, Transformer Fine-Tuning, and DetectGPT
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("📊 COMPREHENSIVE COMPARISON OF AI TEXT DETECTION APPROACHES")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# SECTION 1: LOAD ALL RESULTS
# ============================================================================

print("📁 Loading results from all approaches...\n")

results_files = {
    'Feature-Based ML': {
        'csv': 'feature_ml_comparison.csv',
        'json': 'feature_ml_results.json',
        'label': 'Feature ML'
    },
    'Transformer Fine-Tuning': {
        'csv': 'model_comparison_results.csv',
        'json': 'evaluation_results.json',
        'label': 'Transformer'
    },
    'DetectGPT': {
        'csv': 'detectgpt_comparison.csv',
        'json': 'detectgpt_results.json',
        'label': 'DetectGPT'
    }
}

loaded_results = {}
detailed_info = {}

for approach, files in results_files.items():
    try:
        # Load CSV
        if os.path.exists(files['csv']):
            df = pd.read_csv(files['csv'])
            loaded_results[approach] = df
            print(f"✅ {approach}: Loaded {files['csv']}")
        else:
            print(f"⚠️  {approach}: {files['csv']} not found")
            continue
        
        # Load JSON details
        if os.path.exists(files['json']):
            with open(files['json'], 'r') as f:
                detailed_info[approach] = json.load(f)
            print(f"   └─ Details from {files['json']}")
    except Exception as e:
        print(f"❌ {approach}: Error loading - {e}")

print(f"\n✅ Successfully loaded {len(loaded_results)}/3 approaches\n")

if len(loaded_results) == 0:
    print("❌ No results found! Please run the experiments first.")
    exit()

# ============================================================================
# SECTION 2: EXTRACT BEST RESULTS FROM EACH APPROACH
# ============================================================================

print("🔍 Extracting best results from each approach...\n")

comparison_data = []

# Feature-Based ML - Get best model
if 'Feature-Based ML' in loaded_results:
    df = loaded_results['Feature-Based ML']
    best_idx = df['F1 (Weighted)'].idxmax()
    best_row = df.loc[best_idx]
    
    model_name = best_row.get('Model', 'Unknown')
    
    comparison_data.append({
        'Approach': 'Feature-Based ML',
        'Model/Method': model_name,
        'Accuracy': best_row['Accuracy'],
        'F1 (Weighted)': best_row['F1 (Weighted)'],
        'F1 (Human)': best_row.get('F1 (Human)', 0),
        'F1 (AI)': best_row.get('F1 (AI)', 0),
        'Precision': best_row['Precision'],
        'Recall': best_row['Recall'],
        'AUC-ROC': best_row['AUC-ROC'],
        'PR-AUC': best_row.get('PR-AUC', 0),
        'Train Time (s)': best_row.get('Train Time (s)', 0),
        'Type': 'Classical ML'
    })
    
    print(f"✅ Feature-Based ML: Best model = {model_name}")
    print(f"   F1: {best_row['F1 (Weighted)']:.4f}, Accuracy: {best_row['Accuracy']:.4f}")

# Transformer Fine-Tuning
if 'Transformer Fine-Tuning' in loaded_results:
    df = loaded_results['Transformer Fine-Tuning']
    
    model_name = df['Model'].iloc[0] if 'Model' in df.columns else 'Unknown'
    
    comparison_data.append({
        'Approach': 'Transformer',
        'Model/Method': model_name,
        'Accuracy': df['Accuracy'].iloc[0],
        'F1 (Weighted)': df['F1 (Weighted)'].iloc[0],
        'F1 (Human)': df.get('F1 (Human)', pd.Series([0])).iloc[0],
        'F1 (AI)': df.get('F1 (AI)', pd.Series([0])).iloc[0],
        'Precision': df['Precision'].iloc[0],
        'Recall': df['Recall'].iloc[0],
        'AUC-ROC': df['AUC-ROC'].iloc[0],
        'PR-AUC': df.get('PR-AUC', pd.Series([0])).iloc[0],
        'Train Time (s)': df.get('Train Time (s)', pd.Series([0])).iloc[0],
        'Type': 'Deep Learning'
    })
    
    print(f"✅ Transformer: Model = {model_name}")
    print(f"   F1: {df['F1 (Weighted)'].iloc[0]:.4f}, Accuracy: {df['Accuracy'].iloc[0]:.4f}")

# DetectGPT - Get best method
if 'DetectGPT' in loaded_results:
    df = loaded_results['DetectGPT']
    best_idx = df['F1 (Weighted)'].idxmax()
    best_row = df.loc[best_idx]
    
    method_name = best_row.get('Method', 'DetectGPT')
    
    comparison_data.append({
        'Approach': 'DetectGPT',
        'Model/Method': method_name,
        'Accuracy': best_row['Accuracy'],
        'F1 (Weighted)': best_row['F1 (Weighted)'],
        'F1 (Human)': best_row.get('F1 (Human)', 0),
        'F1 (AI)': best_row.get('F1 (AI)', 0),
        'Precision': best_row['Precision'],
        'Recall': best_row['Recall'],
        'AUC-ROC': best_row['AUC-ROC'],
        'PR-AUC': best_row.get('PR-AUC', 0),
        'Train Time (s)': 0,  # DetectGPT is zero-shot, no training
        'Type': 'Zero-Shot'
    })
    
    print(f"✅ DetectGPT: Method = {method_name}")
    print(f"   F1: {best_row['F1 (Weighted)']:.4f}, Accuracy: {best_row['Accuracy']:.4f}")

# Create comparison DataFrame
final_comparison = pd.DataFrame(comparison_data)

# ============================================================================
# SECTION 3: RESULTS SUMMARY TABLE
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL RESULTS SUMMARY TABLE")
print("="*80)
print()

# Display formatted table
display_cols = ['Approach', 'Model/Method', 'Accuracy', 'F1 (Weighted)', 
                'Precision', 'Recall', 'AUC-ROC', 'Type']
print(final_comparison[display_cols].to_string(index=False))

# Save to CSV
final_comparison.to_csv('FINAL_COMPARISON_ALL_APPROACHES.csv', index=False)
print(f"\n💾 Saved to: FINAL_COMPARISON_ALL_APPROACHES.csv")

# ============================================================================
# SECTION 4: IDENTIFY BEST APPROACH
# ============================================================================

print("\n" + "="*80)
print("🏆 BEST PERFORMERS")
print("="*80)

best_accuracy = final_comparison.loc[final_comparison['Accuracy'].idxmax()]
best_f1 = final_comparison.loc[final_comparison['F1 (Weighted)'].idxmax()]
best_auc = final_comparison.loc[final_comparison['AUC-ROC'].idxmax()]

print(f"\n🥇 Best Accuracy: {best_accuracy['Approach']} ({best_accuracy['Model/Method']})")
print(f"   Score: {best_accuracy['Accuracy']:.4f}")

print(f"\n🥇 Best F1 Score: {best_f1['Approach']} ({best_f1['Model/Method']})")
print(f"   Score: {best_f1['F1 (Weighted)']:.4f}")

print(f"\n🥇 Best AUC-ROC: {best_auc['Approach']} ({best_auc['Model/Method']})")
print(f"   Score: {best_auc['AUC-ROC']:.4f}")

# Calculate average rank
metrics = ['Accuracy', 'F1 (Weighted)', 'Precision', 'Recall', 'AUC-ROC']
ranks = {}
for metric in metrics:
    ranked = final_comparison.sort_values(metric, ascending=False).reset_index(drop=True)
    for idx, row in ranked.iterrows():
        approach = row['Approach']
        if approach not in ranks:
            ranks[approach] = []
        ranks[approach].append(idx + 1)

avg_ranks = {k: np.mean(v) for k, v in ranks.items()}
best_overall = min(avg_ranks, key=avg_ranks.get)

print(f"\n🏆 Best Overall (Average Rank): {best_overall}")
print(f"   Average Rank: {avg_ranks[best_overall]:.2f}")

# ============================================================================
# SECTION 5: STATISTICAL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("📈 STATISTICAL ANALYSIS")
print("="*80)

print("\n1️⃣ Performance Metrics Statistics:")
print(f"   Mean Accuracy: {final_comparison['Accuracy'].mean():.4f} (±{final_comparison['Accuracy'].std():.4f})")
print(f"   Mean F1 Score: {final_comparison['F1 (Weighted)'].mean():.4f} (±{final_comparison['F1 (Weighted)'].std():.4f})")
print(f"   Mean AUC-ROC: {final_comparison['AUC-ROC'].mean():.4f} (±{final_comparison['AUC-ROC'].std():.4f})")

print("\n2️⃣ Performance Gaps:")
max_f1 = final_comparison['F1 (Weighted)'].max()
min_f1 = final_comparison['F1 (Weighted)'].min()
print(f"   F1 Score Range: {min_f1:.4f} to {max_f1:.4f} (gap: {max_f1-min_f1:.4f})")

max_acc = final_comparison['Accuracy'].max()
min_acc = final_comparison['Accuracy'].min()
print(f"   Accuracy Range: {min_acc:.4f} to {max_acc:.4f} (gap: {max_acc-min_acc:.4f})")

print("\n3️⃣ Efficiency Analysis:")
if 'Train Time (s)' in final_comparison.columns:
    for _, row in final_comparison.iterrows():
        time_min = row['Train Time (s)'] / 60
        if time_min > 0:
            print(f"   {row['Approach']}: {time_min:.1f} minutes")
        else:
            print(f"   {row['Approach']}: No training required (zero-shot)")

# ============================================================================
# SECTION 6: COMPREHENSIVE VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("🎨 Creating comprehensive visualizations...")
print("="*80)

# Create large figure with multiple subplots
fig = plt.figure(figsize=(20, 14))

# Color scheme
colors = {'Feature-Based ML': '#3498db', 'Transformer': '#e74c3c', 'DetectGPT': '#2ecc71'}
approach_colors = [colors.get(app, '#95a5a6') for app in final_comparison['Approach']]

# ============================================================================
# Plot 1: Main Metrics Comparison (Bar Chart)
# ============================================================================
ax1 = plt.subplot(3, 3, 1)
metrics_to_plot = ['Accuracy', 'F1 (Weighted)', 'Precision', 'Recall']
x = np.arange(len(final_comparison))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    offset = (i - len(metrics_to_plot)/2 + 0.5) * width
    bars = ax1.bar(x + offset, final_comparison[metric], width, label=metric, alpha=0.8)

ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
ax1.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(final_comparison['Approach'], rotation=15, ha='right')
ax1.legend(loc='lower right', fontsize=9)
ax1.set_ylim([0, 1.1])
ax1.grid(axis='y', alpha=0.3)

# ============================================================================
# Plot 2: F1 Score Comparison (Horizontal Bar)
# ============================================================================
ax2 = plt.subplot(3, 3, 2)
bars = ax2.barh(final_comparison['Approach'], final_comparison['F1 (Weighted)'], 
                color=approach_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('F1 Score (Weighted)', fontsize=11, fontweight='bold')
ax2.set_title('F1 Score Ranking', fontsize=12, fontweight='bold')
ax2.set_xlim([0, 1.1])
ax2.grid(axis='x', alpha=0.3)

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)

# ============================================================================
# Plot 3: AUC-ROC Comparison
# ============================================================================
ax3 = plt.subplot(3, 3, 3)
bars = ax3.bar(final_comparison['Approach'], final_comparison['AUC-ROC'], 
               color=approach_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('AUC-ROC Score', fontsize=11, fontweight='bold')
ax3.set_title('AUC-ROC Comparison', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 1.1])
ax3.grid(axis='y', alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, height + 0.02,
             f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# ============================================================================
# Plot 4: Precision vs Recall
# ============================================================================
ax4 = plt.subplot(3, 3, 4)
for i, row in final_comparison.iterrows():
    ax4.scatter(row['Recall'], row['Precision'], 
                s=300, c=colors.get(row['Approach'], '#95a5a6'), 
                alpha=0.7, edgecolors='black', linewidth=2,
                label=row['Approach'])
    ax4.annotate(row['Approach'], (row['Recall'], row['Precision']),
                 xytext=(5, 5), textcoords='offset points', fontsize=9)

ax4.set_xlabel('Recall', fontsize=11, fontweight='bold')
ax4.set_ylabel('Precision', fontsize=11, fontweight='bold')
ax4.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.set_xlim([0, 1.05])
ax4.set_ylim([0, 1.05])
ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Balance')

# ============================================================================
# Plot 5: Heatmap of All Metrics
# ============================================================================
ax5 = plt.subplot(3, 3, 5)
heatmap_metrics = ['Accuracy', 'F1 (Weighted)', 'Precision', 'Recall', 'AUC-ROC']
heatmap_data = final_comparison[heatmap_metrics].T

sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn',
            xticklabels=final_comparison['Approach'],
            yticklabels=heatmap_metrics, cbar_kws={'label': 'Score'},
            vmin=0.5, vmax=1.0, ax=ax5, linewidths=0.5)
ax5.set_title('Performance Heatmap', fontsize=12, fontweight='bold')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=15, ha='right')

# ============================================================================
# Plot 6: Per-Class F1 Scores
# ============================================================================
ax6 = plt.subplot(3, 3, 6)
x = np.arange(len(final_comparison))
width = 0.35

bars1 = ax6.bar(x - width/2, final_comparison['F1 (Human)'], width, 
                label='Human Class', color='lightblue', edgecolor='black', alpha=0.8)
bars2 = ax6.bar(x + width/2, final_comparison['F1 (AI)'], width, 
                label='AI Class', color='lightcoral', edgecolor='black', alpha=0.8)

ax6.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
ax6.set_title('Per-Class F1 Scores', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(final_comparison['Approach'], rotation=15, ha='right')
ax6.legend()
ax6.set_ylim([0, 1.1])
ax6.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax6.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# ============================================================================
# Plot 7: Radar Chart
# ============================================================================
ax7 = plt.subplot(3, 3, 7, projection='polar')

categories = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC-ROC']
num_vars = len(categories)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

for i, row in final_comparison.iterrows():
    values = [row['Accuracy'], row['F1 (Weighted)'], 
              row['Precision'], row['Recall'], row['AUC-ROC']]
    values += values[:1]
    
    ax7.plot(angles, values, 'o-', linewidth=2, 
             label=row['Approach'], color=colors.get(row['Approach'], '#95a5a6'))
    ax7.fill(angles, values, alpha=0.15, color=colors.get(row['Approach'], '#95a5a6'))

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(categories, fontsize=9)
ax7.set_ylim(0, 1)
ax7.set_title('Overall Performance Radar', fontsize=12, fontweight='bold', pad=20)
ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax7.grid(True)

# ============================================================================
# Plot 8: Approach Type Distribution
# ============================================================================
ax8 = plt.subplot(3, 3, 8)

type_performance = final_comparison.groupby('Type')[['Accuracy', 'F1 (Weighted)', 'AUC-ROC']].mean()
type_performance.plot(kind='bar', ax=ax8, rot=0, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)

ax8.set_ylabel('Average Score', fontsize=11, fontweight='bold')
ax8.set_title('Performance by Approach Type', fontsize=12, fontweight='bold')
ax8.set_xlabel('Approach Type', fontsize=11, fontweight='bold')
ax8.legend(title='Metric', fontsize=9)
ax8.set_ylim([0, 1.1])
ax8.grid(axis='y', alpha=0.3)

# ============================================================================
# Plot 9: Summary Statistics Table
# ============================================================================
ax9 = plt.subplot(3, 3, 9)
ax9.axis('tight')
ax9.axis('off')

summary_stats = []
for metric in ['Accuracy', 'F1 (Weighted)', 'Precision', 'Recall', 'AUC-ROC']:
    best_approach = final_comparison.loc[final_comparison[metric].idxmax(), 'Approach']
    best_score = final_comparison[metric].max()
    mean_score = final_comparison[metric].mean()
    summary_stats.append([metric, best_approach, f"{best_score:.4f}", f"{mean_score:.4f}"])

table = ax9.table(cellText=summary_stats,
                  colLabels=['Metric', 'Best Approach', 'Best Score', 'Mean Score'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color header
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color best scores
for i in range(1, len(summary_stats) + 1):
    for j in range(4):
        if j % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

ax9.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

plt.suptitle('Comprehensive Comparison: AI Text Detection Approaches', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('FINAL_COMPARISON_VISUALIZATIONS.png', dpi=300, bbox_inches='tight')
print("✅ Saved: FINAL_COMPARISON_VISUALIZATIONS.png")
plt.show()

# ============================================================================
# SECTION 7: DETAILED COMPARISON REPORT
# ============================================================================

print("\n" + "="*80)
print("📄 GENERATING DETAILED COMPARISON REPORT")
print("="*80)

report_lines = []
report_lines.append("="*80)
report_lines.append("COMPREHENSIVE COMPARISON REPORT: AI TEXT DETECTION")
report_lines.append("="*80)
report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("")

report_lines.append("1. APPROACHES EVALUATED:")
report_lines.append("-" * 80)
for i, row in final_comparison.iterrows():
    report_lines.append(f"   {i+1}. {row['Approach']} ({row['Type']})")
    report_lines.append(f"      Model/Method: {row['Model/Method']}")
    report_lines.append(f"      F1 Score: {row['F1 (Weighted)']:.4f}")
    report_lines.append("")

report_lines.append("2. PERFORMANCE SUMMARY:")
report_lines.append("-" * 80)
report_lines.append(f"   Overall Best: {best_overall} (Avg Rank: {avg_ranks[best_overall]:.2f})")
report_lines.append(f"   Best Accuracy: {best_accuracy['Approach']} ({best_accuracy['Accuracy']:.4f})")
report_lines.append(f"   Best F1 Score: {best_f1['Approach']} ({best_f1['F1 (Weighted)']:.4f})")
report_lines.append(f"   Best AUC-ROC: {best_auc['Approach']} ({best_auc['AUC-ROC']:.4f})")
report_lines.append("")

report_lines.append("3. DETAILED METRICS:")
report_lines.append("-" * 80)
report_lines.append(final_comparison.to_string(index=False))
report_lines.append("")

report_lines.append("4. KEY INSIGHTS:")
report_lines.append("-" * 80)

# Generate insights
if len(final_comparison) >= 2:
    best_f1_approach = final_comparison.loc[final_comparison['F1 (Weighted)'].idxmax()]
    worst_f1_approach = final_comparison.loc[final_comparison['F1 (Weighted)'].idxmin()]
    gap = best_f1_approach['F1 (Weighted)'] - worst_f1_approach['F1 (Weighted)']
    
    report_lines.append(f"   • Performance gap: {gap:.4f} F1 score difference")
    report_lines.append(f"   • {best_f1_approach['Approach']} outperforms {worst_f1_approach['Approach']} by {gap*100:.2f}%")
    
    # Type analysis
    if 'Type' in final_comparison.columns:
        type_avg = final_comparison.groupby('Type')['F1 (Weighted)'].mean()
        best_type = type_avg.idxmax()
        report_lines.append(f"   • {best_type} approaches perform best on average")
    
    # Balance analysis
    for _, row in final_comparison.iterrows():
        balance = abs(row['Precision'] - row['Recall'])
        if balance < 0.05:
            report_lines.append(f"   • {row['Approach']} shows excellent precision-recall balance")

report_lines.append("")
report_lines.append("5. RECOMMENDATIONS:")
report_lines.append("-" * 80)
report_lines.append(f"   • For best accuracy: Use {best_accuracy['Approach']}")
report_lines.append(f"   • For balanced performance: Use {best_f1['Approach']}")
report_lines.append(f"   • For ROC performance: Use {best_auc['Approach']}")

if 'Train Time (s)' in final_comparison.columns:
    fastest = final_comparison[final_comparison['Train Time (s)'] > 0].nsmallest(1, 'Train Time (s)')
    if not fastest.empty:
        report_lines.append(f"   • For speed: Use {fastest.iloc[0]['Approach']}")

report_lines.append("")
report_lines.append("="*80)
report_lines.append("END OF REPORT")
report_lines.append("="*80)

# Save report
report_text = "\n".join(report_lines)
with open('FINAL_COMPARISON_REPORT.txt', 'w') as f:
    f.write(report_text)

print("✅ Saved: FINAL_COMPARISON_REPORT.txt")
print("\n" + report_text)

# ============================================================================
# SECTION 8: EXPORT SUMMARY
# ============================================================================

print("\n" + "="*80)
print("💾 FILES GENERATED:")
print("="*80)
print("   1. FINAL_COMPARISON_ALL_APPROACHES.csv - Complete metrics table")
print("   2. FINAL_COMPARISON_VISUALIZATIONS.png - All visualizations")
print("   3. FINAL_COMPARISON_REPORT.txt - Detailed text report")

print("\n" + "="*80)
print("✨ COMPARISON COMPLETE!")
print("="*80)
print(f"\n🏆 WINNER: {best_overall}")
print(f"   Best Overall Performance (Average Rank: {avg_ranks[best_overall]:.2f})")
print(f"   F1 Score: {final_comparison[final_comparison['Approach']==best_overall]['F1 (Weighted)'].iloc[0]:.4f}")
print(f"   Accuracy: {final_comparison[final_comparison['Approach']==best_overall]['Accuracy'].iloc[0]:.4f}")
print(f"   AUC-ROC: {final_comparison[final_comparison['Approach']==best_overall]['AUC-ROC'].iloc[0]:.4f}")
print("\n" + "="*80)