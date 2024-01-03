# energy_models
utilities for modeling building system energy usage as a function of weather

## X and Y variables

The `Var()` class is instantiated for X, Y, and y. 

*`Y`: Upper-case Y means 'Y actual.' The variable `Y.pred` then means the actual Y values during the prediction period (to be used to compare against `y.pred` for model scoring).

*`y`: Lower-case y means 'y predicted.'

*`X`: X is upper-case because it represents actual data (and not a model prediction), similar to upper-case Y.