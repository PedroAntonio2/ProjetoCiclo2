from mlflow.tracking import MlflowClient

registered_models = ['MLP', 'DecisionTree', 'KNN', 'RandomForest', 'Bagging', 'GradientBoosting', 'XGBoost', 'LightGBM', 'OLA', 'LCA', 'KNORAU', 'KNORAE', 'MCB']

client = MlflowClient()

model_details = []

for i, model_name in enumerate(registered_models):
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        
        for version in versions:
            model_details.append({
                "Model Name": model_name,
                "Version": version.version,
                "Description": version.description,
            })

            run_id = version.run_id
            run = client.get_run(run_id)
            metrics = run.data.metrics
            parameters = run.data.params

            model_details[-1]["Accuracy"] = metrics.get("accuracy", "N/A")
            model_details[-1]["recall"] = metrics.get("recall", "N/A")
            model_details[-1]['Parameters'] = parameters
            print(model_details[-1])


    except Exception as e:
        print(f"{model_name} Failed: {e}")