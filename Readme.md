# matetools

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
* `advancepvs.py`: advances a number of plies in the given PVs to create new puzzles
* `deducepvs.py`: uses proven PVs, and the associated PVs for all
  the positions along the mating lines, to find possibly missing PVs
* `mergepvs.py`: merges several EPD file containing PVs into one
* `provepvs.py`: uses conjectured PVs to guide a local engine to find mates and
  prove PVs (the scripts works sequentially, proven PVs are available
  immediately)
* `sortbymates.py`: sorts the positions in an EPD file

By way of example, the following EPD files are provided:

* `ChestUCI_23102018.epd`: the original suite derived from publicly available
`ChestUCI.epd` files, see
[FishCooking](https://groups.google.com/g/fishcooking/c/lh1jTS4U9LU/m/zrvoYQZUCQAJ). It contains 6561 positions, with one draw, some illegal positions and some
positions with a sub-optimal value of `bm`.
* `matetrack.epd`: The successor to `ChestUCI_23102018.epd`, with all illegal positions removed and all known errors corrected. In 26 positions the side to move is going to get mated.
* `matetrackpv.epd`: The same as `matetrack.epd` but with PVs leading to the checkmate where such a PV is known.
* `matedtrackpv.epd`: Derived from `matetrackpv.epd` (using the script `advancepvs.py`) by advancing one ply in all positions with `bm>1` that have a PV. In 6502 positions the side to move is going to get mated.
* `matedtrack.epd`: The same as `matedtrackpv.epd`, but with the PV information removed.

### Automatic creation of new test positions

With the help of the script `advancepvs.py` it is easy to derive new mate
puzzles from the information stored in `matetrackpv.epd`. For example, the file `matedtrack.epd` has been created with the command
```shell
python advancepvs.py --plies 1 --mateType won && sed 's/; PV.*/;/' matedtrackpv.epd > matedtrack.epd
```

---
## Related repositories

* [vondele/matetrack](https://github.com/vondele/matetrack)
* [robertnurnberg/cdbmatetrack](https://github.com/robertnurnberg/cdbmatetrack) 

---
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
