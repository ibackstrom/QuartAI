#!/bin/sh
set -eu
python -m qbench.data
python -m unittest qllm.tests.test_layers
python -m unittest tests.test_models
python -m qbench.verify --device auto
python -m qbench.benchmark
python -m qbench.run_crossover
python -m qbench.analysis.analyze
python -m qbench.run_ablations
python -m qbench.generate
python -m qbench.analysis.analyze
python -m qbench.analysis.publish
