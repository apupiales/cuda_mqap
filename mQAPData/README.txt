mQAP benchmark data (third-party)
=================================

The *.dat and *.PO files in this folder are NOT part of the cuda_mqap source code and are NOT
covered by its GPL v3 license. They are third-party benchmark data, redistributed here unmodified
for academic and research use, with attribution to their authors.


Origin
------

Instances (*.dat)
    Test suite of the multiobjective Quadratic Assignment Problem (mQAP) by Joshua Knowles and
    David Corne, generated with their instance generators makeQAPuni.cc and makeQAPrl.cc
    ((C) Joshua Knowles, 2002).
    Original page (no longer online):
        http://www.cs.bham.ac.uk/~jdk/mQAP/
    Archived copy (last modified July 2003):
        https://web.archive.org/web/2019/http://www.cs.bham.ac.uk/~jdk/mQAP/

Pareto optimal fronts (*.PO)
    Enumeration of the Pareto optima of the ten-facility instances, provided by Gary Lamont and
    published on the same page (corrected version).

Copy used by this project
    The files were taken from https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData
    (added by Fred Sundvik in 2015). They are identical to that copy except for line endings.
    That repository states an MIT license for its tests folder, but it does not own the data.

    KC30-2fl-1rl.dat is not listed on the original page; it was probably generated later with the
    same generator (makeQAPrl). Its provenance could not be confirmed.


Terms of use
------------

No explicit license is published for the instances or the .PO files. The original page describes
the generators as "free software provided for academic or educational use. If used for commercial
purposes, please contact me" (Joshua Knowles). Use this data for academic and research purposes
and cite the publication below; for commercial use, contact the authors.


Citation
--------

J. D. Knowles and D. W. Corne. Instance Generators and Test Suites for the Multiobjective
Quadratic Assignment Problem. In Evolutionary Multi-Criterion Optimization (EMO 2003),
Lecture Notes in Computer Science, vol. 2632, pp. 295-310. Springer, 2003.

@InProceedings{Knowles2003mQAP,
  author    = {Joshua D. Knowles and David W. Corne},
  title     = {Instance Generators and Test Suites for the Multiobjective Quadratic Assignment Problem},
  booktitle = {Evolutionary Multi-Criterion Optimization (EMO 2003)},
  series    = {Lecture Notes in Computer Science},
  volume    = {2632},
  pages     = {295--310},
  publisher = {Springer},
  year      = {2003}
}


File formats
------------

KC<n>-<k>fl-<type>.dat
    Header line (either "facilities = n objectives = k ..." or "facilities: n objectives: k ..."),
    followed by the n x n distance matrix and k flow matrices of n x n.
    <type>: "rl" = real-like instances, "uni" = uniform instances.

KC10-2fl-<type>.PO
    One Pareto optimal solution per line: a 1-based permutation (location of each facility)
    followed by its k objective values.
