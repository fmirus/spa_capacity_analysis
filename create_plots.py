import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def superposition_plot(df, save_path):

    sp = np.unique(df.combinations)
    dim_ranges = np.unique(df.dimension)

    sns.set(style="whitegrid")
    sns.set(font_scale=1.5)
    fp = sns.catplot(x='combinations',
                     y='similarity',
                     hue='val',
                     col='dimension',
                     kind='box',
                     data=df,
                     height=30,
                     aspect=1,
                     legend=False,
                     whis=[0.05, 0.95])
    fp.set_axis_labels('# Superpositions', 'Similarity')
    labels = sp
    labels = [label if label %20 ==0 else '' for label in labels]
    fp.set_xticklabels(labels)
    for ind, (ax, dim) in enumerate(zip(fp.axes.flat, dim_ranges)):
        ax.grid(ls=':')
        ax.axhline(2./np.sqrt(dim),
                   c='r',
                   ls=':',
                   label='weak similarity threshold')
        ax.axhline(3./np.sqrt(dim),
                   c='g',
                   ls=':',
                   label='strong similarity threshold')
        ax.set_title(label='Dimension = %i'%dim)
        ax.set_xticklabels(ax.get_xticklabels())
        ax.set_yticklabels(np.round(np.arange(0, 1.1, 0.1), 1), fontsize=16)
        ax.set_xlabel('# Superpositions')
        if dim_ranges[ind] == dim_ranges[0]:
            ax.set_ylabel('Similarity')
        if dim_ranges[ind] == dim_ranges[-1]:
            ax.legend(loc=1, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)

if not os.path.exists('plots'):
    os.makedirs('plots')

df = pd.read_hdf('data/superpos_saturation.h5')
superposition_plot(df, save_path='plots/spa_superposition_capacity.eps')

df = pd.read_hdf('data/spa_power_saturation_analysis.h5')

sns.set_style("whitegrid")
sns.set(font_scale=1.5)
fp = sns.factorplot(x='num_superpos',
                    y='similarity',
                    hue='val',
                    col='dimension',
                    kind='box',
                    data=df,
                    height=30,
                    aspect=1,
                    legend=False)

fp.set_axis_labels('# Superpositions', 'Similarity')

dim_ranges = np.unique(df.dimension)

for ind, (ax, dim) in enumerate(zip(fp.axes.flat, dim_ranges)):
    ax.grid(ls=':')
    ax.axhline(2./np.sqrt(dim),
               c='r',
               ls=':',
               label='weak similarity threshold')
    ax.axhline(3./np.sqrt(dim),
               c='g',
               ls=':',
               label='strong similarity threshold')

    ax.set_title(label='Dimension = %i'%dim)
    ax.set_xticklabels(ax.get_xticklabels())
    ax.set_xlabel('# Superpositions')

    if dim_ranges[ind] == dim_ranges[0]:
        ax.set_ylabel('Similarity')

    if dim_ranges[ind] == dim_ranges[-1]:
        ax.legend(loc='best', frameon=True)
plt.tight_layout()
plt.savefig('plots/spa_power_capacity.eps', dpi=1200)

fp = sns.factorplot(x='max_items_per_class',
                    y='similarity',
                    hue='val',
                    col='dimension',
                    kind='box',
                    data=df,
                    height=30,
                    aspect=1,
                    legend=False)

fp.set_axis_labels('# Superpositions', 'Similarity')

dim_ranges = np.unique(df.dimension)

for ind, (ax, dim) in enumerate(zip(fp.axes.flat, dim_ranges)):
    ax.grid(ls=':')
    ax.axhline(2./np.sqrt(dim),
               c='r',
               ls=':',
               label='weak similarity threshold')

    ax.axhline(3./np.sqrt(dim),
               c='g',
               ls=':',
               label='strong similarity threshold')

    ax.set_title(label='Dimension = %i'%dim)
    ax.set_xticklabels(ax.get_xticklabels())
    ax.set_xlabel('# Superpositions per class')

    if dim_ranges[ind] == dim_ranges[0]:
        ax.set_ylabel('Similarity')

    if dim_ranges[ind] == dim_ranges[-1]:
        ax.legend(loc='best', frameon=True)

plt.tight_layout()
plt.savefig('plots/spa_power_capacity_superpositions_per_class.eps', dpi=1200)
