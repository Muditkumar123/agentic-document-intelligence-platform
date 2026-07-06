"""Runtime monitoring: query logging and drift detection.

The evaluation gates prove quality on the golden distribution; drift detection
answers the follow-up question production teams actually face — *are live
queries still the distribution those numbers were measured on?* When they are
not, the measured quality claims silently stop applying, and that is exactly
the signal these reports surface.
"""
