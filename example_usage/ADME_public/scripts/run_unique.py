from unique import Pipeline
import pandas as pd

# CLint Rat
pipeline = Pipeline.from_config("../unique_config_files/rLM LogCLint_unique_config_file.yaml")
uq_methods_outputs, uq_evaluation_outputs = pipeline.fit()

pd.DataFrame.from_dict(uq_methods_outputs).to_csv(pipeline.output_path / "clint_rat_uq_metrics_values.csv", index=False)

# CLint Human
pipeline = Pipeline.from_config("../unique_config_files/hLM LogCLint_unique_config_file.yaml")
uq_methods_outputs, uq_evaluation_outputs = pipeline.fit()

pd.DataFrame.from_dict(uq_methods_outputs).to_csv(pipeline.output_path / "clint_human_uq_metrics_values.csv", index=False)

# PPB Rat
pipeline = Pipeline.from_config("../unique_config_files/LogFu-Rat_unique_config_file.yaml")
uq_methods_outputs, uq_evaluation_outputs = pipeline.fit()

pd.DataFrame.from_dict(uq_methods_outputs).to_csv(pipeline.output_path / "ppb_rat_uq_metrics_values.csv", index=False)

# PPB Human
pipeline = Pipeline.from_config("../unique_config_files/LogFu-Human_unique_config_file.yaml")
uq_methods_outputs, uq_evaluation_outputs = pipeline.fit()

pd.DataFrame.from_dict(uq_methods_outputs).to_csv(pipeline.output_path / "ppb_human_uq_metrics_values.csv", index=False)

# MDCK-MDR1 ER
pipeline = Pipeline.from_config("../unique_config_files/MDCK-MDR1_LogER_unique_config_file.yaml")
uq_methods_outputs, uq_evaluation_outputs = pipeline.fit()

pd.DataFrame.from_dict(uq_methods_outputs).to_csv(pipeline.output_path / "mdck_mdr1_er_uq_metrics_values.csv", index=False)
