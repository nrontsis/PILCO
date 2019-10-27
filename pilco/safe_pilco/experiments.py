from safe_cars_run import safe_cars


name = "cars"
for i in range(5):
    safe_cars(name=name+str(i), seed=i, logging=True)
