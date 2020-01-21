import numpy as np
import nengo_spa as spa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_similarity_df(dim, n_comb, operation, seed, vocab):
    """calculate a pandas data frame with similarities for each of
    the dimension over a certain number of samples

    :dim: the vector dimensions to be evaluated
    :n_comb: the number of combinations (items to combine using the operation)
    :operation: 'superpos' or 'binding'
    :seed: random number seed
    :vocab: vector vocabulary
    :returns: a pandas dataframe containing all similarities

    """
    data = []
    d = np.zeros(dim)
    if operation != 'superpos':
        d[0] = 1
    superpos = spa.SemanticPointer(d)

    sims = np.zeros(2*n_comb)

    for v in np.arange(2*n_comb):
        if v < n_comb:
            k = 'C%s'%str(v).zfill(len(str(2*n_comb)))
            if operation == 'superpos':
                superpos += vocab[k]
            else:
                superpos = superpos*vocab[k]

    for v in np.arange(2*n_comb):
        k = 'C%s'%str(v).zfill(len(str(2*n_comb)))
        sims[v] = np.abs(superpos.compare(vocab[k]))

        val = 'member'
        if v >= n_comb:
            val = 'no member'

        data.append(dict(dimension=dim,
                         similarity=sims[v],
                         combinations=n_comb,
                         operation=operation,
                         val=val,
                         seed=seed,
                         sample=v))

    return pd.DataFrame(data)


def create_random_vocab(dim, num_vector_items, seed, b_unitary=False):
    """create a random spa vocab with a certain number of vectors

    :dim: dimension of the vectors
    :num_vector_items: number of vectors in the Vocabulary
    :seed: random number seed
    :b_unitary: bool indication if the created vector shall be unitary
    :returns: an instance of a spa vocabulary

    """
    vocab = spa.Vocabulary(dimensions=dim,
                           pointer_gen=np.random.RandomState(seed=seed))

    for i in np.arange(num_vector_items):
        k = 'C%s'%str(i).zfill(len(str(num_vector_items)))

        if b_unitary:
            vocab.populate(k+'.unitary()')
        else:
            vocab.populate(k)

    return vocab


def plot(df, sp, dim_ranges):
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


if __name__ == "__main__":
    dim_list = [256, 512, 1024]
    num_comb_list = np.arange(10, 210, 10)
    operation = 'superpos'
    seeds = np.arange(3)
    b_calculate=False
    if b_calculate:
        first = True
        for seed in seeds:
            for dim in dim_list:
                for num_combs in num_comb_list:
                    vocab = create_random_vocab(dim, 2*num_combs, seed=seed)
                    if first:
                        df = get_similarity_df(dim=dim,
                                               n_comb=num_combs,
                                               operation=operation,
                                               seed=seed,
                                               vocab=vocab)
                        first = False
                    else:
                        df = df.append(get_similarity_df(dim=dim,
                                                         n_comb=num_combs,
                                                         operation=operation,
                                                         seed=seed,
                                                         vocab=vocab))

        df.to_hdf('data/superpos_saturation.h5', key='df')
    else:
        df = pd.read_hdf('data/superpos_saturation.h5')
        plot(df, num_comb_list, dim_list)
        plt.show()
