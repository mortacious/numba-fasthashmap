import operator

import numba as nb
import numpy as np
from numba.core import types
from numba.experimental import structref
from numba.extending import overload, overload_method, overload_attribute


__all__ = ["FastHashMap"]


@structref.register
class FastHashMapType(types.StructRef):
    def preprocess_fields(self, fields):
        # We don't want the struct to take Literal types.
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class _FastHashMap(structref.StructRefProxy):
    def __new__(cls, mem, load_faction, hash_fn):
        size = 0

        if load_faction > 0.95 or load_faction < 0.25:
            raise ValueError("Invalid load faction.")

        cap = mem.shape[0] - 1
        remap_threshold = cap * load_faction
        zero_key = False

        return structref.StructRefProxy.__new__(cls,
                                                mem,
                                                size,
                                                load_faction,
                                                remap_threshold,
                                                zero_key,
                                                hash_fn)

    def __len__(self):
        return _FastHashMap_len(self)

    @property
    def capacity(self):
        return _FastHashMap_capacity(self)

    def __getitem__(self, key):
        return _FastHashMap_getitem(self, key)

    def __setitem__(self, key, value):
        _FastHashMap_setitem(self, key, value)

    def __delitem__(self, key):
        _FastHashMap_delitem(self, key)

    def __array__(self):
        return _FastHashMap_array(self)

    def clear(self):
        _FastHashMap_clear(self)

    def keys(self):
        return _FastHashMap_keys(self)

    def values(self):
        return _FastHashMap_values(self)

    def items(self):
        for it in _FastHashMap_items(self):
            yield it["key"], it["value"]

    def __iter__(self):
        yield from self.keys()

    def __getnewargs__(self):
        return _FastHashMap_newargs(self)

    def __getstate__(self):
        return _FastHashMap_getstate(self)

    def __setstate__(self, state):
        _FastHashMap_setstate(self, state)


structref.define_proxy(_FastHashMap, FastHashMapType,
                       ["mem", "size", "load_faction", "remap_threshold", "zero_key", "hash_fn"])


@nb.njit()
def _FastHashMap_len(self):
    return len(self)


@nb.njit()
def _FastHashMap_capacity(self):
    return self.capacity


@nb.njit()
def _FastHashMap_getitem(self, key):
    return self[key]


@nb.njit()
def _FastHashMap_setitem(self, key, value):
    self[key] = value


@nb.njit()
def _FastHashMap_delitem(self, key):
    del self[key]


@nb.njit()
def _FastHashMap_array(self):
    return self.mem


@nb.njit()
def _FastHashMap_clear(self):
    self.clear()


@nb.njit()
def _FastHashMap_keys(self):
    return self.keys()


@nb.njit()
def _FastHashMap_values(self):
    return self.values()


@nb.njit()
def _FastHashMap_items(self):
    return self.items()


@nb.njit()
def _FastHashMap_newargs(self):
    return self.mem, self.load_faction, self.hash_fn


@nb.njit()
def _FastHashMap_getstate(self):
    return self.size, self.remap_threshold, self.zero_key


@nb.njit()
def _FastHashMap_setstate(self, state):
    self.size, self.remap_threshold, self.zero_key, = state


@nb.njit(nogil=True, fastmath=True, inline='always')
def default_hash(key):
    return key.__hash__()


def FastHashMap(key_dtype, value_dtype, hash_fn=None, initial_capactiy=32, load_faction=0.75):
    item_dtype = np.dtype([("key", key_dtype), ("value", value_dtype)])
    mem = np.zeros(initial_capactiy + 1, dtype=item_dtype)

    if hash_fn is None:
        hash_fn = default_hash

    hm = _FastHashMap(mem, load_faction, hash_fn)
    return hm


@overload(len, jit_options={'nogil': True, 'cache': True})
def _fasthashmap_len(hm):
    if isinstance(hm, FastHashMapType):
        def len_impl(hm):
            return hm.size
        return len_impl


@overload_attribute(FastHashMapType, "capacity", jit_options={'nogil': True, 'cache': True})
def _fasthashmap_capacity(self):
    def get(self):
        return self.mem.shape[0] - 1
    return get


@overload_attribute(FastHashMapType, "mem_view", jit_options={'nogil': True, 'cache': True})
def _fasthashmap_mem_view(self):
    def mem_view(self):
        return self.mem.view(np.uint8).reshape(self.capacity + 1, -1)

    return mem_view


@overload_method(FastHashMapType, "clear", jit_options={'nogil': True, 'cache': True})
def _fasthashmap_clear(hm):
    def clear_impl(hm):
        if hm.size > 0:
            hm.mem = np.zeros_like(hm.mem)
            hm.zero_key = False
            hm.size = 0

    return clear_impl


@overload_method(FastHashMapType, "remap", jit_options={'nogil': True, 'cache': True})
def _fasthashmap_remap(hm):
    def remap_impl(hm):
        if hm.size < hm.remap_threshold:
            return

        cap = hm.capacity
        new_mem = np.zeros(cap*2 + 1, dtype=hm.mem.dtype)
        mod = hm.mem.shape[0] - 2

        for i in range(cap):
            if hm.mem[i]["key"]:
                pos = hm.hash_fn(hm.mem[i]["key"]) & mod

                while True:
                    if not new_mem[pos]["key"]:
                        new_mem[pos] = hm.mem[i]
                        break
                    pos = (pos + 1) & mod

        # copy the last element over into the new array
        new_mem[-1] = hm.mem[-1]
        hm.mem = new_mem
        hm.remap_threshold = cap * 2 * hm.load_faction

    return remap_impl


@overload(operator.getitem, jit_options={'nogil': True, 'cache': True})
def _fasthashmap_getitem(hm, key):
    if isinstance(hm, FastHashMapType):
        def get_impl(hm, key):

            if not key:
                if hm.zero_key:
                    return hm.mem[-1]["value"]
                else:
                    raise KeyError("Key not found")

            mod = hm.mem.shape[0] - 2

            h = hm.hash_fn(key)
            pos = h & mod

            while True:
                if not hm.mem[pos]["key"]:
                    raise KeyError("Key not found")
                elif not key == hm.mem[pos]["key"]:
                    pos = (pos + 1) & mod
                    continue

                return hm.mem[pos]["value"]

        return get_impl


@overload(operator.setitem, jit_options={'nogil': True, 'cache': True})
def _fasthashmap_setitem(hm, key, value):
    if isinstance(hm, FastHashMapType):
        def set_impl(hm, key, value):
            # attempt a remap first
            hm.remap()

            if not key:
                # special handling of zero keys
                if not hm.zero_key:
                    hm.size += 1
                hm.zero_key = True
                hm.mem[-1]["value"] = value
                return

            mod = hm.mem.shape[0] - 2
            h = hm.hash_fn(key)
            pos = h & mod

            while True:
                if not hm.mem[pos]["key"]:
                    hm.size += 1

                elif key != hm.mem[pos]["key"]:
                    pos = (pos + 1) & mod
                    continue

                hm.mem[pos]["key"] = key
                hm.mem[pos]["value"] = value
                return

        return set_impl


@overload(operator.delitem, jit_options={'nogil': True, 'cache': True})
def _fasthashmap_del(hm, key):
    def del_impl(hm, key):
        if not key:
            if hm.zero_key:
                hm.zero_key = False
                hm.size -= 1
                return
            else:
                raise KeyError("Key not found")
        mod = hm.mem.shape[0] - 2

        h = hm.hash_fn(key)
        pos = h & mod
        mem_view = hm.mem_view
        while True:
            if not hm.mem[pos]["key"]:
                raise KeyError("Key not found")
            elif not key == hm.mem[pos]["key"]:
                pos = (pos + 1) & mod
                continue

            hm.size -= 1
            mem_view[pos] = 0

            prev = pos
            it = pos

            while True:
                it = (it + 1) & mod
                if not hm.mem[it]["key"]:
                    break

                p = hm.hash_fn(hm.mem[it]["key"]) & mod
                if p > it and (p <= prev or it >= prev) or (p <= prev <= it):
                    mem_view[prev] = mem_view[it]
                    mem_view[it] = 0
                    prev = it
            return

    return del_impl


@overload_method(FastHashMapType, "keys", jit_options={'nogil': True, 'cache': True})
def _fasthashmap_keys(hm):
    def keys_impl(hm):
        keys = np.empty(hm.size, dtype=hm.mem["key"].dtype)
        index = 0
        if hm.zero_key:
            keys[index] = hm.mem[-1]["key"]
            index += 1
        for i in range(hm.capacity):
            k = hm.mem[i]["key"]
            if k:
                keys[index] = k
                index += 1

        return keys
    return keys_impl


@overload_method(FastHashMapType, "values", jit_options={'nogil': True, 'cache': True})
def _fasthashmap_values(hm):
    def values_impl(hm):
        values = np.empty(hm.size, dtype=hm.mem["value"].dtype)
        index = 0
        if hm.zero_key:
            values[index] = hm.mem[-1]["value"]
            index += 1
        for i in range(hm.capacity):
            if hm.mem[i]["key"]:
                values[index] = hm.mem[i]["value"]
                index += 1

        return values
    return values_impl


@overload_method(FastHashMapType, "items", jit_options={'nogil': True, 'cache': True})
def _fasthashmap_values(hm):
    def items_impl(hm):
        items = np.empty(hm.size, dtype=hm.mem.dtype)
        index = 0
        if hm.zero_key:
            items[index] = hm.mem[-1]
            index += 1
        for i in range(hm.capacity):
            if hm.mem[i]["key"]:
                items[index] = hm.mem[i]
                index += 1

        return items
    return items_impl










