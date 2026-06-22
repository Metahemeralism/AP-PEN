"""
Real commodity futures data loading and preprocessing.

TODO: Implement loaders for real futures data:
      - CME crude oil / natural gas term-structure CSV/parquet
      - Align maturities to standard tenors
      - Compute log-price observations for GS Kalman filter input
      - Emit a dataset dict in the same shape as data/generate.py
        (log-futures panel + maturities + dates), minus the synthetic
        'delta_true' ground truth which real data does not have
"""
