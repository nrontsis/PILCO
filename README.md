A branch to check whether caching the factorizations makes sense.

I don't think that it does, as the speedup is never more than x2. Usually it's much less as the dataset increases, and the cost is dominated by the gradient calculations.

I believe that tensorflow avoids recalculating the factorizations across the horizon. Each time we evaluate the cost of a controller we calculate the factorization (only) once.

I haven't tried on sparse gps but I don't see why they should be different.

Check `examples/time_factorizations.py`