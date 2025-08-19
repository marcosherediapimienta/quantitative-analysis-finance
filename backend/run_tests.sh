#!/bin/bash
source ../.venv/bin/activate
python -m unittest discover backend/option_pricing/tests
