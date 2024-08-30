# Usage

Finally, once the data and configuration files have been prepared, you can run `UNIQUE` in the following way:

```python
from unique import Pipeline

# Prepare UNIQUE pipeline
pipeline = Pipeline.from_config("/path/to/config.yaml")

# Run UNIQUE pipeline
uq_methods_outputs, uq_evaluation_outputs = pipeline.fit()
# Returns: (Dict[str, np.ndarray], Dict[str, pd.DataFrame])
```

Fitting the `Pipeline` will return two dictionaries:

|`uq_methods_outputs`|`uq_evaluation_outputs`|
|:-------------------|:----------------------|
|Contains each computed UQ method's name (as in `"UQ_Method_Name[Input_Name(s)]"`) and corresponding UQ values.|Contains, for each evaluation benchmark, the evaluation metrics for all the computed UQ methods (organized in one `pd.DataFrame` for each evaluation benchmark).|

Additionally, `UNIQUE` also generates graphical outputs in the form of tables and evaluation plots (if `display_outputs` is enabled and the code is running in a JupyterNotebook cell).

:::{seealso}
Check out the [Examples](../examples/index.md) to see `UNIQUE` in action.
:::
