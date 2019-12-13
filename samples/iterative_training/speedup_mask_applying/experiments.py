import numpy as np

from samples.iterative_training.Timer import timeit


def main():
    maskFile = "f_7926_528400.00_528.40_masks.npy"
    masks = np.load(maskFile)
    print(masks.shape, masks.dtype)
    with timeit():
        agg1 = np.sum(masks, axis=-1)
    print(agg1.shape, agg1.dtype)

    n = masks.shape[-1]
    agg2 = masks[..., 0]
    with timeit():
        for i in range(1, n):
            np.logical_or(agg2, masks[..., i], out=agg2)

    np.testing.assert_array_equal(agg1.astype(np.bool), agg2)

    with timeit():
        agg3 = np.any(masks, axis=-1)
    np.testing.assert_array_equal(agg2, agg2)


main()
