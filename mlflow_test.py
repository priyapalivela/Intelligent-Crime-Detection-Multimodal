import mlflow
mlflow.set_tracking_uri(
    "file:///C:/Users/TRN Srinivas/Desktop/"
    "Intelligent-Crime-Detection-Multimodal/mlruns"
)

# Log your ACTUAL project metrics
mlflow.set_experiment("Crime_Detection_Experiment")

with mlflow.start_run(run_name="CNN_BiLSTM_DistilBERT_Fusion"):

    # YOUR actual architecture parameters
    mlflow.log_param("audio_encoder",    "CNN-BiLSTM")
    mlflow.log_param("text_encoder",     "DistilBERT-base-uncased")
    mlflow.log_param("fusion_strategy",  "conservative_max")
    mlflow.log_param("embedding_dim",    256)
    mlflow.log_param("audio_dataset",    "UrbanSound8K_8732clips")
    mlflow.log_param("text_dataset",     "Chicago_PD_IUCR")
    mlflow.log_param("loss_function",    "SupervisedContrastiveLoss")
    mlflow.log_param("optimizer",        "AdamW")
    mlflow.log_param("scheduler",        "CosineAnnealingLR")
    mlflow.log_param("batch_size",       32)
    mlflow.log_param("epochs",           50)
    mlflow.log_param("deployment",       "HuggingFace_Spaces_Docker")

    # YOUR actual results
    mlflow.log_metric("accuracy",        0.8829)
    mlflow.log_metric("weighted_f1",     0.86)
    mlflow.log_metric("f2_score",        0.88)

    # Ablation study results
    mlflow.log_metric("audio_only_acc",  0.74)
    mlflow.log_metric("text_only_acc",   0.81)
    mlflow.log_metric("fusion_acc",      0.8829)

    mlflow.set_tag("project", "Intelligent_Crime_Detection")
    mlflow.set_tag("version", "v1.3.0")

    print("Your project metrics logged to MLflow!")