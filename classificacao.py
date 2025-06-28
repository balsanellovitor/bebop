import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    matthews_corrcoef, cohen_kappa_score, 
    log_loss, accuracy_score, confusion_matrix, 
    roc_curve, auc
)
import matplotlib.pyplot as plt


# ========== Configurações ==========
USE_BALANCAMENTO = True

feature_base_folder = "C://Users//admin//Desktop//all_features_fluo"
experiment_base_folder = "C://Users//admin//Desktop//experimentos_fluo//7 - melanoma x nev_sek"
models = ["swin", "vit", "tnt", "deit", "cait", "dino"]

# "CCE", "BCC", "MEL" == Maligno
# "NEV", "SEK", "ACK" == Benigno

classe_1 = ["MEL"]
classe_0 = ["SEK", "NEV"]

# ========== Funções Auxiliares ==========

def load_data(feature_folder, class_groups):
    X, y = [], []
    for class_name, label in class_groups.items():
        feature_file = os.path.join(feature_folder, f"{class_name}_features_{os.path.basename(feature_folder)}.npy")
        label_file = os.path.join(feature_folder, f"{class_name}_labels_{os.path.basename(feature_folder)}.npy")

        if os.path.exists(feature_file) and os.path.exists(label_file):
            features = np.load(feature_file)
            labels = np.load(label_file)
            print(f"Carregado {len(features)} imagens da classe {class_name} ({label})")
            y.extend([label] * len(features))
            X.extend(features)
    return np.array(X), np.array(y)


def normalize_z_max(X, scaler=None):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X = (X - mean) / std
    if scaler is None:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, scaler


def balance_with_kmeans(X, y):
    print("Executando balanceamento com K-Means...")
    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_samples = max(class_counts)
    X_balanced, y_balanced = [], []

    for cls in unique_classes:
        X_class = X[y == int(cls)]
        y_class = y[y == int(cls)]
        n_samples = len(X_class)

        if n_samples < max_samples:
            n_clusters = n_samples
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_class)

            samples_per_cluster = max_samples // n_clusters
            for cluster in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster)[0]
                selected_indices = np.random.choice(cluster_indices, min(samples_per_cluster, len(cluster_indices)), replace=True)
                X_balanced.extend(X_class[selected_indices])
                y_balanced.extend(y_class[selected_indices])
        else:
            X_balanced.extend(X_class)
            y_balanced.extend(y_class)

    return np.array(X_balanced), np.array(y_balanced)


# ========== Execução Principal ==========

class_groups = {cls: 1 for cls in classe_1}
class_groups.update({cls: 0 for cls in classe_0})

for model in models:
    feature_folder = os.path.join(feature_base_folder, model)
    save_path = os.path.join(experiment_base_folder, model, "resultados_CLASSIFICADORES")
    os.makedirs(save_path, exist_ok=True)

    print(f"\nCarregando características de: {feature_folder}")
    X, y = load_data(feature_folder, class_groups)
    if X.size == 0 or y.size == 0:
        print(f"Aviso: Nenhum dado encontrado para o modelo {model}. Pulando para o próximo.")
        continue

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        "SVM": SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }

    for clf_name, clf_proto in classifiers.items():
        print(f"\nAvaliando {clf_name} com cross-validation para {model}...")

        accuracies, baccs, aucs = [], [], []

        tprs = []
        aucs_roc = []
        mean_fpr = np.linspace(0, 1, 100)

        plt.figure(figsize=(8, 6))

        precisions, recalls, specificities = [], [], []
        f1s, mccs, kappas, loglosses = [], [], [], []

        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"  Fold {fold_idx + 1}/5...")

            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            X_train_fold, scaler = normalize_z_max(X_train_fold)
            X_test_fold, _ = normalize_z_max(X_test_fold, scaler)

            if USE_BALANCAMENTO:
                X_train_fold, y_train_fold = balance_with_kmeans(X_train_fold, y_train_fold)

            clf = clf_proto.__class__(**clf_proto.get_params())
            clf.fit(X_train_fold, y_train_fold)

            y_pred = clf.predict(X_test_fold)
            y_proba = clf.predict_proba(X_test_fold)[:, 1] if hasattr(clf, "predict_proba") else np.zeros_like(y_pred)

            fpr, tpr, _ = roc_curve(y_test_fold, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"Fold {fold_idx + 1} (AUC = {roc_auc:.2f})")

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs_roc.append(roc_auc)

            tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred).ravel()

            accuracy = accuracy_score(y_test_fold, y_pred)
            bacc = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))

            precision = precision_score(y_test_fold, y_pred, zero_division=0)
            recall = recall_score(y_test_fold, y_pred, zero_division=0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = f1_score(y_test_fold, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test_fold, y_pred)
            kappa = cohen_kappa_score(y_test_fold, y_pred)
            logloss = log_loss(y_test_fold, y_proba) if not np.all(y_proba == 0) else np.nan

            accuracies.append(accuracy)
            baccs.append(bacc)
            aucs.append(roc_auc)
            precisions.append(precision)
            recalls.append(recall)
            specificities.append(specificity)
            f1s.append(f1)
            mccs.append(mcc)
            kappas.append(kappa)
            loglosses.append(logloss)

        # Plot ROC Média
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs_roc)

        plt.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Curva ROC Média (AUC = %0.2f ± %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

        plt.fill_between(
            mean_fpr,
            tpr_lower,
            tpr_upper,
            color="grey",
            alpha=0.2,
            label=r"± 1 STD",
        )

        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Média - {clf_name} ({model})")
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(save_path, f"roc_curve_media_{clf_name}.png"))
        plt.close()

        # Relatório final
        avg_accuracy = np.mean(accuracies)
        avg_bacc = np.mean(baccs)
        avg_auc = np.mean(aucs)

        with open(os.path.join(save_path, f"relatorio_cv_{clf_name}.txt"), "w") as f:
            f.write(f"Resultados Cross-Validation ({clf_name} - {model}):\n")
            f.write(f"Acurácia média: {avg_accuracy:.4f}\n")
            f.write(f"Acurácia Balanceada média: {avg_bacc:.4f}\n")
            f.write(f"AUC-ROC média: {avg_auc:.4f}\n")

        df_folds = pd.DataFrame({
            'Fold': [1, 2, 3, 4, 5],
            'Accuracy': accuracies,
            'Balanced_Accuracy': baccs,
            'AUC': aucs,
            'Precision': precisions,
            'Recall': recalls,
            'Specificity': specificities,
            'F1_Score': f1s,
            'MCC': mccs,
            'Cohen_Kappa': kappas,
            'Log_Loss': loglosses
        })
        df_folds.to_csv(os.path.join(save_path, f"resultados_folds_{clf_name}.csv"), index=False)

print("\n✅ Treinamento, curvas ROC e resultados dos folds salvos para todos os modelos!")
