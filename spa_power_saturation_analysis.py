import numpy as np
import nengo_spa as spa
import pandas as pd


class findCombinationsClass(object):
    """class to find all possible combinations of positive integers
    that sum up to a given number"""

    def __init__(self, sum_number):
        """class initialization

        :sum_number: positive number all found combinations should sum up to

        """
        self._sum_number = sum_number

    def findCombinationsUtil(self, arr, index, num, reducedNum):
        # Base condition
        if (reducedNum < 0):
            return

        # If combination is
        # found, print it
        if (reducedNum == 0):
            self._result.append(arr[:index])
            return

        # Find the previous number stored in arr[].
        # It helps in maintaining increasing order
        prev = 1 if(index == 0) else arr[index - 1]

        # note loop starts from previous
        # number i.e. at array location
        # index - 1
        for k in range(prev, num + 1):

            # next element of array is k
            arr[index] = k

            # call recursively with
            # reduced number
            self.findCombinationsUtil(arr, index + 1, num, reducedNum - k)

    def findCombinations(self):
        self._result = []
        self.arr = [0] * self._sum_number

        # array to store the combinations
        # It can contain max n elements

        # find all combinations
        self.findCombinationsUtil(self.arr,
                                  0,
                                  self._sum_number,
                                  self._sum_number)
        return self._result


def power(s, e):
    x = np.fft.ifft(np.fft.fft(s.v) ** e).real
    return spa.SemanticPointer(data=x)


def create_random_vocab(dim,
                        num_vector_items,
                        seed,
                        b_unitary=False,
                        spatial_ids=['X', 'Y']):
    """create a random spa vocab with a certain number of vectors

    :dim: dimension of the vectors
    :num_vector_items: number of vectors in the Vocabulary
    :seed: random number seed
    :b_unitary: bool indication if the created vector shall be unitary
    :spatial_ids: list of ids for the spatial vectors, which shall be unitary
    :returns: an instance of a spa vocabulary

    """

    vocab = spa.Vocabulary(dimensions=dim,
                           pointer_gen=np.random.RandomState(seed=seed))

    for k in spatial_ids:
        vocab.populate(k+'.unitary()')

    for i in np.arange(num_vector_items):
        k = 'C%s'%str(i).zfill(len(str(num_vector_items)))

        if b_unitary:
            vocab.populate(k+'.unitary()')
        else:
            vocab.populate(k)

    return vocab


def generate_positions(num_items, seed=None):
    """function to randomly generate x and y
    coordinates to be encoded in vectors

    :num_items: number of positions
    :seed: seed for random generator
    :returns: lists of x and y corrdinates

    """
    if seed is not None:
        np.random.seed(seed)
    xs = np.round(np.random.uniform(-0.99, 1, num_items)*10, 3)
    ys = np.round(np.random.uniform(-0.99, 1, num_items)*10, 3)

    return xs, ys


def create_superposition_vector(vocab, xs, ys, items_per_class_list):
    """generate superposition vector

    :vocab: instance of a nengo_spa vocabulary
    :xs: x coordinates of the positions
    :ys: y corrdinates of the positions
    :items_per_class_list: list of number that indicates how often one
                           particular class occurs in the superposition vector
    :returns: superposition vector as nengo_spa.SemanticPointer object

    """
    lens = np.array([len(xs), len(ys), np.sum(items_per_class_list)])
    if ~np.all(lens[0] == lens):
        raise ValueError('input array have different lengths')

    num = len(xs)
    sup = spa.SemanticPointer(np.zeros(vocab.dimensions))
    class_ind = 0
    added = np.zeros(len(items_per_class_list))

    for i, (x, y) in enumerate(zip(xs, ys)):
        sup += vocab['C%s'%str(class_ind).zfill(len(str(num)))] * \
            power(vocab['X'], x) * power(vocab['Y'], y)
        added[class_ind] += 1
        if added[class_ind] == items_per_class_list[class_ind]:
            class_ind += 1

    return sup


def get_similarity_data(vec,
                        items_per_class_list,
                        vocab,
                        xs,
                        ys,
                        seed,
                        epsilon=0.4,
                        limit=10,
                        num_test_positions=100,
                        data=[]):
    dim = vocab.dimensions

    xx = np.linspace(-limit, limit, num_test_positions)
    yy = np.linspace(-limit, limit, num_test_positions)

    class_ind = 0
    added = np.zeros(len(items_per_class_list))
    num = len(xs)
    member_sims=[]
    no_member_sims=[]

    for i, (x_gt, y_gt) in enumerate(zip(xs, ys)):
        # print('class ind %i'%class_ind)
        ck = 'C%s'%str(class_ind).zfill(len(str(num)))

        added[class_ind] += 1
        if added[class_ind] == items_per_class_list[class_ind]:
            class_ind += 1

        for i, x in enumerate(xx):
            for j, y in enumerate(yy):
                pos_v = power(vocab['X'], x)*power(vocab['Y'], y)
                cx = np.isclose(x_gt, x, atol=epsilon, rtol=0)
                cy = np.isclose(y_gt, y, atol=epsilon, rtol=0)
                sim = None
                if np.any(np.logical_and(cx, cy)):
                    # print('positive x=%1.3f y=%1.3f'%(x,y))
                    sim = pos_v.compare(vec*vocab[ck].__invert__())
                    member_sims.append(sim)
                else:
                    sim = pos_v.compare(vec*vocab[ck].__invert__())
                    no_member_sims.append(sim)

    data.append(dict(dimension=dim,
                     similarity=np.mean(member_sims),
                     val='member',
                     seed=seed,
                     num_superpos=num,
                     num_classes=len(items_per_class_list),
                     min_items_per_class=np.min(items_per_class_list),
                     max_items_per_class=np.max(items_per_class_list)
                     ))

    data.append(dict(dimension=dim,
                     similarity=np.mean(np.abs(no_member_sims)),
                     val='no_member',
                     seed=seed,
                     num_superpos=num,
                     num_classes=len(items_per_class_list),
                     min_items_per_class=np.min(items_per_class_list),
                     max_items_per_class=np.max(items_per_class_list)
                     ))

    data.append(dict(dimension=dim,
                     similarity=np.percentile(np.abs(no_member_sims), 5),
                     val='no_member',
                     seed=seed,
                     num_superpos=num,
                     num_classes=len(items_per_class_list),
                     min_items_per_class=np.min(items_per_class_list),
                     max_items_per_class=np.max(items_per_class_list)
                     ))

    data.append(dict(dimension=dim,
                     similarity=np.percentile(np.abs(no_member_sims), 95),
                     val='no_member',
                     seed=seed,
                     num_superpos=num,
                     num_classes=len(items_per_class_list),
                     min_items_per_class=np.min(items_per_class_list),
                     max_items_per_class=np.max(items_per_class_list)
                     ))
    return data


def calculate_similarity_df(seeds, num_sups_list, dim_list, export_path=None):
    """function to calculate the similarities between the superposition
    vectors and vectors encoding test positions

    :seeds: list of random number seeds
    :num_sups_list: list containing the number of
                    superpositions to be evaluated
    :dim_list: list containing the vector dimensions to be evaluated
    :export_path: path to to a .h5 file to export the resulting dataframe to.
                  If no such path is specified the function
                  returns the calculated DataFrame
    :returns: a pandas DataFrame (only if no export_path is specified)

    """
    data = []
    for num_sups in num_sups_list:
        fcc = findCombinationsClass(sum_number=num_sups)
        combs = fcc.findCombinations()
        if (num_sups - 1)%5 == 0:
            df = pd.DataFrame(data)
            df.to_hdf('data/spa_power_sat_df_intermediate_%i.h5'%(num_sups-1),
                      key='df')
            del(df)

        for i, seed in enumerate(seeds):

            xs, ys = generate_positions(num_items=num_sups, seed=seed)

            for dim in dim_list:
                print('-------------------------------------')
                print('number of superpositions=%i, \
                      dimension=%i, seed=%i'%(num_sups, dim, seed))
                vocab = create_random_vocab(dim=dim,
                                            num_vector_items=num_sups,
                                            seed=seed)
                sup_vecs = [create_superposition_vector(vocab,
                    xs,
                    ys,
                    items_per_class_list=items_p_cl) for items_p_cl in combs]

                for i in np.arange(len(sup_vecs)):
                    data = get_similarity_data(vec=sup_vecs[i],
                                               items_per_class_list=combs[i],
                                               vocab=vocab,
                                               xs=xs,
                                               ys=ys,
                                               seed=seed,
                                               epsilon=0.4,
                                               limit=10,
                                               num_test_positions=100,
                                               data=data)

    df = pd.DataFrame(data)

    if export_path is not None:
        df.to_hdf(export_path, key='df')
    else:
        return df


if __name__ == "__main__":
    # seeds = np.arange(3)
    seeds = [0]
    dim_list = [256, 512, 1024]
    num_sups_list = np.arange(5)+1

    export_path = 'data/spa_power_saturation_analysis_df.h5'
    calculate_similarity_df(seeds, num_sups_list, dim_list, export_path)
