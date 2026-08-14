# Raw pipeline output

This directory is empty in the repository: the Propp-fr output for the full
corpus (`.book` and `.entities`, one pair per novel) is far too large to
distribute here.

It is needed only to rebuild the derived tables with
`scripts/00_build_networks.py`. Every analysis in the paper reads the derived
tables in `../derived/`, which are included.

Expected layout:

    data/raw/gallica/1863_Feval_Paul_Les-Habits-noirs.book
    data/raw/gallica/1863_Feval_Paul_Les-Habits-noirs.entities
    ...
