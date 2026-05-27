import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

experiment = client.get_experiment_by_name('vehicules_mlops')

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=['metrics.accuracy DESC'],
    max_results=1
)

best_run = runs[0]

best_run_id = best_run.info.run_id

model_uri = f'runs:/{best_run_id}/model'

registered = mlflow.register_model(
    model_uri=model_uri,
    name='vehicule_prediction_model'
)

client.update_registered_model(
    name='vehicule_prediction_model',
    description='Modèle ML de prédiction de véhicules'
)

client.set_model_version_tag(
    name='vehicule_prediction_model',
    version=registered.version,
    key='validated_by',
    value='sarra_team'
)

client.transition_model_version_stage(
    name='vehicule_prediction_model',
    version=registered.version,
    stage='Staging'
)

acc = best_run.data.metrics['accuracy']

SEUIL = 0.85

if acc >= SEUIL:

    client.transition_model_version_stage(
        name='vehicule_prediction_model',
        version=registered.version,
        stage='Production'
    )

    print('Modèle promu en Production')

else:
    print('Accuracy insuffisante')