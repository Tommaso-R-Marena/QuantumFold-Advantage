#!/bin/bash
# Script to restore the production notebook
git show 2384cad:examples/complete_production_run.ipynb > examples/complete_production_run.ipynb
git add examples/complete_production_run.ipynb  
git commit -m "Restore complete production notebook from 2384cad"
git push
