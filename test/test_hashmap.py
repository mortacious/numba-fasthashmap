import pytest
from numba_fasthashmap import FastHashMap
import numpy as np
import numba as nb
import time
from numba.typed import Dict


@nb.njit(nogil=True)
def hash_fn(key):
    return key


def test_hashmap_create():
    key_dtype = np.int32
    value_dtype = np.float32
    hm = FastHashMap(key_dtype, value_dtype, hash_fn, initial_capactiy=32)

    assert len(hm) == 0
    assert hm.capacity == 32

    key = np.int32(1)
    value = np.float32(13.4)

    hm[key] = value

    key2 = np.int32(33)
    value2 = np.float32(15.4)
    hm[key2] = value2

    assert hm[key] == value
    assert len(hm) == 2

    del hm[key]

    with pytest.raises(KeyError):
        _ = hm[key]
    assert len(hm) == 1
    assert np.asarray(hm)[1]["key"] == 33

    hm[0] = 10.1
    assert len(hm) == 2

    hm[0] = 111
    assert np.asarray(hm)[-1]["value"] == 111

    cnt = 0
    for k, v in hm.items():
        cnt += 1
    assert cnt == 2


def test_hashmap_default_hash():
    key_dtype = np.int32
    value_dtype = np.float32
    hm = FastHashMap(key_dtype, value_dtype, initial_capactiy=32)

    assert len(hm) == 0
    assert hm.capacity == 32

    key = np.int32(1)
    value = np.float32(13.4)

    hm[key] = value

    key2 = np.int32(33)
    value2 = np.float32(15.4)
    hm[key2] = value2

    assert hm[key] == value
    assert len(hm) == 2
    print(np.asarray(hm))

    del hm[key]

    with pytest.raises(KeyError):
        _ = hm[key]
    assert len(hm) == 1
    assert np.asarray(hm)[1]["key"] == 33

    hm[0] = 10.1
    assert len(hm) == 2

    hm[0] = 111
    assert np.asarray(hm)[-1]["value"] == 111


def test_hashmap_insert():
    key_dtype = np.int32
    value_dtype = np.float32
    hm = FastHashMap(key_dtype, value_dtype, initial_capactiy=32)

    values = np.random.random(10000)
    for i in range(values.shape[0]):
        hm[i] = values[i]

    assert len(hm) == values.shape[0]

    map_vals = hm.values()

    assert np.allclose(sorted(map_vals), sorted(values))


def test_hashmap_insert_numba():
    key_dtype = np.int32
    value_dtype = np.float32
    hm = FastHashMap(key_dtype, value_dtype, initial_capactiy=32)

    values = np.random.random(50_000_000)

    @nb.njit(nogil=True)
    def insert_values(hm, values):
        for i in range(values.shape[0]):
            hm[i] = values[i]

    tic = time.time()
    insert_values(hm, values)
    print("warmup FastHashMap insert took", time.time()-tic)
    hm = FastHashMap(key_dtype, value_dtype, initial_capactiy=32)

    tic = time.time()
    insert_values(hm, values)
    print("compiled FastHashMap insert took", time.time()-tic)
    assert len(hm) == values.shape[0]

    numba_dict = Dict.empty(key_type=nb.from_dtype(key_dtype), value_type=nb.from_dtype(value_dtype))

    @nb.njit(nogil=True)
    def dict_insert_values(nbd, values):
        for i in range(values.shape[0]):
            nbd[i] = values[i]

    tic = time.time()
    dict_insert_values(numba_dict, values)
    print("warmup numba.typed.Dict insert took", time.time()-tic)
    numba_dict = Dict.empty(key_type=nb.from_dtype(key_dtype), value_type=nb.from_dtype(value_dtype))

    tic = time.time()
    dict_insert_values(numba_dict, values)
    print("compiled numba.typed.Dict insert took", time.time()-tic)
    assert len(numba_dict) == values.shape[0]


def test_hashmap_pickle():
    import pickle
    key_dtype = np.int32
    value_dtype = np.float32
    hm = FastHashMap(key_dtype, value_dtype, initial_capactiy=32)

    values = np.random.random(1000)
    for i in range(values.shape[0]):
        hm[i] = values[i]

    hm_pickled = pickle.dumps(hm)

    hm_unpickled = pickle.loads(hm_pickled)
    assert len(hm) == values.shape[0]

    map_vals = hm.values()

    assert np.allclose(sorted(map_vals), sorted(values))