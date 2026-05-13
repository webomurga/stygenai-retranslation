import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from math import pi

# Set universal styling for academic plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
CLASS_NAMES = ["Human", "NMT", "LLM"]  
COLORS = {"Human": "#1f77b4", "NMT": "#ff7f0e", "LLM": "#2ca02c"}

def force_string_labels(y_array):
    """Bulletproof mapping of any numeric label format to our class strings."""
    label_map = {0: "Human", 1: "NMT", 2: "LLM", "0": "Human", "1": "NMT", "2": "LLM"}
    return [label_map.get(lbl, "Unknown") for lbl in y_array]

def plot_confusion_matrix(y_true, y_pred, output_path="output/confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Number of Segments'})
    
    plt.title('Random Forest Confusion Matrix', pad=15, fontsize=14, fontweight='bold')
    plt.ylabel('True Translation Source', fontweight='bold')
    plt.xlabel('Predicted Translation Source', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_importance(rf_model, feature_names, top_n=15, output_path="output/feature_importance.png"):
    plt.figure(figsize=(10, 8))
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    top_features_raw = [str(feature_names[i]) for i in indices]
    top_importances = importances[indices]
    
    # Format labels and assign colors: Quotes for keywords, normal text for stylistic features
    display_features = []
    palette = []
    
    for feature in top_features_raw:
        # Check if it's a keyword (typically all lowercase or just digits)
        if feature.islower() or feature.isdigit():
            display_features.append(f'"{feature}"')  # Add quotation marks
            palette.append('#9467bd')                # Purple for Keyword
        else:
            display_features.append(feature)         # Keep as is
            palette.append('#17becf')                # Blue for Morpho-Stylistic

    sns.barplot(x=top_importances, y=display_features, palette=palette)
    plt.title(f'Top {top_n} Most Discriminative Features', pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('Mean Decrease in Impurity (Importance Score)')
    plt.ylabel('Feature (Keyword or Stylistic Marker)')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#17becf', label='Morpho-Stylistic'),
                       Patch(facecolor='#9467bd', label='Keyword')]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return top_features_raw # Return raw features so the Violin plot can still find them in the dataframe!

def plot_tsne_clusters(X, y, output_path="output/tsne_clusters.png"):
    plt.figure(figsize=(10, 8))
    X_scaled = StandardScaler().fit_transform(X)
    
    # NEW: Enhanced t-SNE parameters for better visual separation
    # - early_exaggeration: Pushes distinct clusters further apart from each other (default is 12)
    # - init='pca': Creates a more stable and readable global layout
    # - n_iter: Gives the algorithm more time to optimize the spacing
    tsne = TSNE(
        n_components=2, 
        perplexity=min(30, len(X)-1), 
        early_exaggeration=30.0, 
        max_iter=2000,
        init='pca',
        random_state=42
    )
    X_tsne = tsne.fit_transform(X_scaled)
    
    y_labels = force_string_labels(y)
    
    # Increased marker size (s=120) and slight opacity tweak for readability
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_labels, hue_order=CLASS_NAMES,
                    palette=COLORS, style=y_labels, s=120, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    plt.title('t-SNE Visualization of Stylistic Chunk Clusters', pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Source', frameon=True, shadow=True)
    
    # NEW: Dynamically scale the axes to give the clusters "room to breathe"
    x_min, x_max = X_tsne[:, 0].min(), X_tsne[:, 0].max()
    y_min, y_max = X_tsne[:, 1].min(), X_tsne[:, 1].max()
    padding_x = (x_max - x_min) * 0.1
    padding_y = (y_max - y_min) * 0.1
    plt.xlim(x_min - padding_x, x_max + padding_x)
    plt.ylim(y_min - padding_y, y_max + padding_y)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_stylistic_violins(X, y, feature_names, features_to_plot, output_path="output/violin_plots.png"):
    df = pd.DataFrame(X, columns=feature_names)
    df['Label'] = force_string_labels(y)
    
    # Grab up to the top 4 features that actually exist in the dataframe
    valid_features = [f for f in features_to_plot if f in df.columns][:4]
    
    if not valid_features:
        return

    n_features = len(valid_features)
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 6))
    if n_features == 1: axes = [axes]
    
    for ax, feature in zip(axes, valid_features):
        sns.violinplot(data=df, x='Label', y=feature, palette=COLORS, order=CLASS_NAMES, ax=ax, inner="quartile")
        ax.set_title(feature, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Frequency / Score')

    plt.suptitle('Distribution of Top Discriminative Markers', y=1.05, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_chart(X, y, feature_names, category_mapping, output_path="output/radar_chart.png"):
    """5. Radar (Spider) Chart of Stylistic Profiles (Reverted to StandardScaler with Gridlines)"""
    df = pd.DataFrame(X, columns=feature_names)
    df['Label'] = force_string_labels(y)
    
    # Reverting to StandardScaler for a more natural proportion
    scaler = StandardScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names])
    
    profile_data = {'Label': CLASS_NAMES}
    for category, features in category_mapping.items():
        valid_feats = [f for f in features if f in feature_names]
        if valid_feats:
            grouped_means = df.groupby('Label')[valid_feats].mean().mean(axis=1)
            profile_data[category] = grouped_means.reindex(CLASS_NAMES).fillna(0).values
        else:
            profile_data[category] = [0, 0, 0]
            
    radar_df = pd.DataFrame(profile_data)
    categories = list(category_mapping.keys())
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=11, fontweight='bold')
    ax.set_rlabel_position(0)
    
    # --- NEW: Explicitly generate and display inner circles (y-ticks) ---
    numeric_data = radar_df.drop('Label', axis=1).values
    min_val, max_val = np.min(numeric_data), np.max(numeric_data)
    
    # Create 4 or 5 clear concentric circle levels based on the data range
    ticks = np.linspace(np.floor(min_val), np.ceil(max_val), 5)
    ax.set_yticks(ticks)
    # Format the tick labels to be subtle but readable
    ax.set_yticklabels([f"{t:.1f}" for t in ticks], color="grey", size=9)
    
    # Add a little padding to the chart boundaries
    ax.set_ylim(np.floor(min_val) - 0.2, np.ceil(max_val) + 0.2)
    # -------------------------------------------------------------------
    
    for idx, row in radar_df.iterrows():
        values = radar_df.loc[idx].drop('Label').values.flatten().tolist()
        values += values[:1]
        label = row['Label']
        color = COLORS.get(label, "#333333")
        
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=label, color=color)
        ax.fill(angles, values, color=color, alpha=0.15)
        
    plt.title('Normalized Stylistic Profiles', size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()