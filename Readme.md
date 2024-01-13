# Matetools

A collection of Python scripts to manipulate and provide mating PVs for
Chest-like EPD files. Such files store chess mate puzzles in the form
```
<FEN>' bm #'<bm>';'{<comments>}
```
Here `FEN` usually means just the first four fields of a standard FEN, and `bm`
is a nonzero integer that indicates the (currently) fastest known mate for the
given position.

The aim of the scripts in the repo is to convert such collections of puzzles
into the format
```
<FEN>' bm #'<bm>';'{' PV: '<PV>}';'{<comments>}
```
where `PV` is a *proven* PV for the claimed mate.

To this end the following scripts are provided:

* `addpvs.py`: adds (missing) PVs to a given EPD file using a local engine (the
  script uses concurrency, results are available once _all_ positions have been
processed)
* `advancepv.py`: advances a number of plies in the given PVs to create new puzzles
* `mergepvs.py`: merges several EPD file containing PVs into one
* `provepvs.py`: uses conjectured PVs to guide a local engine to find mates and
  prove PVs (the scripts works sequentially, proven PVs are available
immediately)
* `sortmates.py`: sorts the positions in an EPD file

By way of example, the following EPD files are provided:

* `ChestUCI_23102018.epd`: the original suite derived from publicly available
`ChestUCI.epd` files, see
[FishCooking](https://groups.google.com/g/fishcooking/c/lh1jTS4U9LU/m/zrvoYQZUCQAJ). It contains 6561 positions, with some wrongly classified
mates, one draw and some illegal positions.
* `matetrack.epd`: The successor to `ChestUCI_23102018.epd`, with all illegal positions removed and all known errors corrected. In 26 positions the side to move is going to get mated.
* `matedtrack.epd`: Derived from `matetrack.epd` (using the script `advancepv.py`) by advancing one ply in all positions with `bm>1` that have a PV. In 6459 positions the side to move is going to get mated.

---
## Related repositories

* [vondele/matetrack](https://github.com/vondele/matetrack)
* [robertnurnberg/cdbmatetrack](https://github.com/robertnurnberg/cdbmatetrack) ---
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
