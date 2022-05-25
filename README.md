# Numba-Fasthashmap
 
This package provides a simplified and very fast hash map implementation (`FastHashMap`) for numba. 
When compiled, this custom map should be multiple times faster than the default numba.typed.Dict
version, especially for a large number of inserts.

`FastHashMap` uses a regular numpy record array as storage and can therefore be pickled and unpickled as well. 

Note: This package is still in a very early stage of development so there might be unexpected behavior.

## Installation

### Using pip
```
pip install numba-fasthashmap
```

### From source
```
git clone https://github.com/mortacious/numba-fasthashmap.git
cd numba-fasthashmap
python setup.py install
```

### Disclaimer

The implementation is inpired by the pure C99 sc library available [here](https://github.com/tezc/sc).