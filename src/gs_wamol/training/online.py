"""
Online learning with time-decay (Whack 3).

TODO: Implement sequential / streaming calibration where:
      - New market observations arrive at each time step
      - A time-decay schedule down-weights stale residuals
      - The third "whack" re-balances weights across the rolling window
      - Integrates with trainer.py via a callback or iterator interface
"""
