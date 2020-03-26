import numpy as np
import pandas as pd
import nengo_spa as spa
import matplotlib.pyplot as plt
import seaborn as sns
import os


def power(s, e):
    x = np.fft.ifft(np.fft.fft(s.v) ** e).real
    return spa.SemanticPointer(data=x)


def create_vocab(dim=512, seed=2):
    voc = spa.Vocabulary(dimensions=dim)
    voc.populate('BICYCLE; CAR; MOTORCYCLE; PEDESTRIAN; \
                 STATIONARY; TRUCK; UNKNOWN')
    spatial_ids = ['POS_X',
                   'POS_Y',
                   'VEL_X',
                   'VEL_Y',
                   'ACC_X',
                   'ACC_Y',
                   'REL_LANE',
                   'DIST_LEFT_LANE_BORDER',
                   'DIST_RIGHT_LANE_BORDER']

    for k in spatial_ids:
        voc.populate(k+'.unitary()')
    return voc


def heatmap_plot(img,
                 lim,
                 xx,
                 dim=None,
                 gtx=None,
                 gty=None,
                 tidx=None,
                 cmin=0.0,
                 cmax=1.0,
                 filename=None,
                 dpi=1200):
    from matplotlib import cm
    from matplotlib import colors
    label_font_size = 34
    tick_font_size = 28
    legend_font_size =30
    b_gt = False

    if gtx is not None and gty is not None:
        b_gt = True

    plt.figure(figsize=(40, 10))
    plt.subplot(1, 3, 1)
    if tidx is not None:
        plt.title('time exponent: %i'%tidx)
    cax = plt.imshow(img,
                     interpolation='none',
                     extent=(-lim,
                             lim,
                             lim,
                             -lim),
                     cmap=cm.viridis,
                     norm=colors.Normalize(vmin=cmin,
                                           vmax=cmax))
    if b_gt:
        plt.scatter(gtx, gty, facecolors='none', edgecolors='r')

    cbar = plt.colorbar(cax,
                        boundaries=np.linspace(cmin,
                                               cmax,
                                               20,
                                               endpoint=True),
                        ticks=np.round(np.linspace(cmin,
                                                   cmax,
                                                   20,
                                                   endpoint=True),
                                       3))
    cbar.ax.set_ylabel('Similarity',
                       rotation=90,
                       fontsize=label_font_size)
    cbar.ax.tick_params(labelsize=tick_font_size)

    plt.ylabel('y-coordinates', fontsize=label_font_size)
    plt.xlabel('x-coordinates', fontsize=label_font_size)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)

    ax1 = plt.subplot(1, 3, 2)
    ax1.grid(ls=':')
    ax1.set_ylim([cmin, cmax])
    for ind1, _ in enumerate(xx):
        plt.plot(xx, img[ind1, :])
    if dim is not None:
        plt.axhline(2/np.sqrt(dim),
                    ls='--',
                    c='k',
                    alpha=0.5,
                    label='weak similarity threshold')
        plt.axhline(3/np.sqrt(dim),
                    ls='--',
                    c='m',
                    alpha=0.5,
                    label='strong similarity threshold')

    if b_gt:
        for i, gx in enumerate(gtx):
            if i < len(gtx)-1:
                plt.axvline(gx,
                            ls='--',
                            alpha=1.)
            else:
                plt.axvline(gx,
                            ls='--',
                            alpha=1.,
                            label='actual x')

    plt.legend(loc=1, fontsize=legend_font_size)
    plt.ylabel('Similarity', fontsize=label_font_size)
    plt.xlabel('x-coordinates', fontsize=label_font_size)

    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(tick_font_size)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(tick_font_size)

    ax2 = plt.subplot(1, 3, 3)
    ax2.grid(ls=':')
    ax2.set_ylim([cmin, cmax])
    for ind1, _ in enumerate(xx):
        plt.plot(xx, img[:, ind1])
    if dim is not None:
        plt.axhline(2/np.sqrt(dim),
                    ls='--',
                    c='k',
                    alpha=0.5,
                    label='weak similarity threshold')
        plt.axhline(3/np.sqrt(dim),
                    ls='--',
                    c='m',
                    alpha=0.5,
                    label='strong similarity threshold')

    if b_gt:
        for i, gy in enumerate(gty):
            if i < len(gty)-1:
                plt.axvline(gy,
                            ls='--',
                            alpha=1.)
            else:
                plt.axvline(gy,
                            ls='--',
                            alpha=1.,
                            label='actual y')

    plt.legend(loc=1, fontsize=legend_font_size)
    plt.ylabel('Similarity', fontsize=label_font_size)
    plt.xlabel('y-coordinates', fontsize=label_font_size)

    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(tick_font_size)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(tick_font_size)

    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)


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

D=512
voc = create_vocab(D, 3)

ns = 2
xs = np.round(np.random.uniform(-0.99, 1, ns), 3)
ys = np.round(np.random.uniform(-0.99, 1, ns), 3)
ts = np.linspace(0.1, 3, ns)
xs = np.array([-0.526, 0.738])
ys = np.array([-0.68, 0.418])

lim = 10

v = spa.SemanticPointer(np.zeros(D))
w = spa.SemanticPointer(np.zeros(D))
for ind, (t, x, y) in enumerate(zip(ts, xs, ys)):
    if ind < len(xs)//2:
        v += voc['CAR']*power(voc['POS_X'], x*lim)*power(voc['POS_Y'], y*lim)
        w += voc['CAR']*power(voc['POS_X'], x*lim)*power(voc['POS_Y'], y*lim)
    else:
        w += voc['CAR']*power(voc['POS_X'], x*lim)*power(voc['POS_Y'], y*lim)

M = 100
xx = np.linspace(-lim, lim, M)
yy = np.linspace(-lim, lim, M)
ii = np.arange(0, ns)

vs = np.zeros((M, M))
ws = np.zeros((M, M))
for i, x in enumerate(xx):
    for j, y in enumerate(yy):
        test_v = power(voc['POS_X'], x)*power(voc['POS_Y'], y)
        vs[j, i] = test_v.compare(voc['CAR'].__invert__()*v)
        ws[j, i] = test_v.compare(voc['CAR'].__invert__()*w)

heatmap_plot(np.abs(vs),
             lim,
             xx,
             dim=D,
             gtx=xs[:len(xs)//2]*lim,
             gty=ys[:len(ys)//2]*lim,
             tidx=None,
             cmin=0.0,
             cmax=0.7,
             filename='plots/spa_power_representation_one_item.eps')
heatmap_plot(np.abs(ws),
             lim,
             xx,
             dim=D,
             gtx=xs*lim,
             gty=ys*lim,
             tidx=None,
             cmin=0.0,
             cmax=0.7,
             filename='plots/spa_power_representation_two_items.eps')

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
