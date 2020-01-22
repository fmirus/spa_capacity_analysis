import numpy as np
import nengo_spa as spa
import pandas as pd


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


if __name__ == "__main__":
    dim_list = [256, 512, 1024]
    num_comb_list = np.arange(10, 210, 10)
    operation = 'superpos'
    seeds = np.arange(3)
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
