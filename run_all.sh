#!/usr/bin/env bash
# Reproduce every result and figure in the paper.
set -euo pipefail
echo "== 1. trajectory ==";          python3 scripts/01_trajectory.py
echo "== 2. period/genre/canon ==";  python3 scripts/02_drivers_ols.py
echo "== 3. author effects ==";      python3 scripts/03_author_effects.py
echo "== 4. serialization ==";       python3 scripts/04_serialization.py --draws 100000
echo "== 5. specification curve =="; python3 scripts/05_specification_curve.py
echo "== 6. robustness ==";          python3 scripts/06_robustness.py
echo "== 7. genre gradation ==";     python3 scripts/07_genre_gradation.py
echo; echo "figures -> figures/   results -> results/"
