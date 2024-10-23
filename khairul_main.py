import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy.signal import savgol_filter
from scipy.stats import mode
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

# Enable GPU acceleration if available
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Updated Euclidean distance function to avoid data type mismatch
@tf.function
def tf_euclidean_distance(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1))

def load_data(file_path):
    return pd.read_csv(file_path)

def augment_data(data):
    data['Distance'] = np.sqrt(data['X']**2 + data['Y']**2)
    data['Velocity'] = data['Distance'].diff().fillna(0)
    data['Acceleration'] = data['Velocity'].diff().fillna(0)
    data['Jerk'] = data['Acceleration'].diff().fillna(0)
    data['Rolling_Mean'] = data['Distance'].rolling(window=5, min_periods=1).mean()
    data['Angle'] = np.arctan2(data['Y'], data['X'])
    return data

def handle_missing_values(data):
    return data.fillna(data.mean())

def remove_outliers(data):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    preds = iso_forest.fit_predict(data[['X', 'Y', 'Velocity', 'Acceleration', 'Jerk', 'Rolling_Mean']])
    return data[preds != -1]

def smooth_data(data, window_length=5, polyorder=2):
    columns_to_smooth = ['X', 'Y', 'Distance', 'Velocity', 'Acceleration', 'Jerk']
    smoothed_data = Parallel(n_jobs=-1)(delayed(savgol_filter)(data[col], window_length, polyorder) for col in columns_to_smooth)
    for col, smoothed in zip(columns_to_smooth, smoothed_data):
        data[col] = smoothed
    return data

def standardize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=data.columns), scaler

def apply_pca(data, n_components=3):
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data)
    print(f"Explained variance: {sum(pca.explained_variance_ratio_):.2f}")
    return data_reduced, pca

def apply_clustering(data):
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(data)
    dbscan = DBSCAN(eps=0.5, min_samples=5).fit(data)
    agglomerative = AgglomerativeClustering(n_clusters=2).fit(data)
    spectral = SpectralClustering(n_clusters=2, random_state=42).fit(data)
    
    return kmeans.labels_, gmm.predict(data), dbscan.labels_, agglomerative.labels_, spectral.labels_

def ensemble_clustering(kmeans_labels, gmm_labels, dbscan_labels, agglomerative_labels, spectral_labels):
    ensemble_labels = np.array([kmeans_labels, gmm_labels, dbscan_labels, agglomerative_labels, spectral_labels]).T
    return mode(ensemble_labels, axis=1)[0].flatten()

def create_1d_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(2)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_1d_cnn(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    
    # Reshape the input data
    X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    model = create_1d_cnn_model((X_train.shape[1], 1))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, 
                        validation_split=0.2, verbose=1, callbacks=[early_stopping])
    return model, history, X_test, y_test

def evaluate_clustering(data, labels):
    silhouette = silhouette_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    
    print(f"Silhouette Score: {silhouette}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

def calculate_errors(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")

# Updated Visualization with bottom margin fix
def visualize_results(raw_data, processed_data, predictions, ensemble_labels, history):
    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
    
    # Raw vs Processed Data
    axs[0, 0].scatter(raw_data['X'], raw_data['Y'], c='red', alpha=0.5, label='Raw Data', s=20)
    axs[0, 0].scatter(predictions[:, 0], predictions[:, 1], c='blue', alpha=0.5, label='Processed Data', s=20)
    axs[0, 0].set_title('Raw vs Processed Sensor Data', fontsize=16)
    axs[0, 0].set_xlabel('X Coordinate', fontsize=14)
    axs[0, 0].set_ylabel('Y Coordinate', fontsize=14)
    axs[0, 0].legend(fontsize=12)
    axs[0, 0].grid(True)
    
    # Clustering Results
    scatter = axs[0, 1].scatter(processed_data['X'], processed_data['Y'], c=ensemble_labels, cmap='viridis', s=20)
    axs[0, 1].set_title('Clustering Results', fontsize=16)
    axs[0, 1].set_xlabel('X Coordinate', fontsize=14)
    axs[0, 1].set_ylabel('Y Coordinate', fontsize=14)
    axs[0, 1].grid(True)
    fig.colorbar(scatter, ax=axs[0, 1], label='Cluster')
    
    # Error Analysis
    raw_data_tensor = tf.convert_to_tensor(raw_data[['X', 'Y']].values[-len(predictions):], dtype=tf.float32)
    processed_data_tensor = tf.convert_to_tensor(processed_data[['X', 'Y']].values[-len(predictions):], dtype=tf.float32)
    predictions_tensor = tf.convert_to_tensor(predictions, dtype=tf.float32)
    
    raw_error = tf_euclidean_distance(raw_data_tensor, predictions_tensor).numpy()
    processed_error = tf_euclidean_distance(processed_data_tensor, predictions_tensor).numpy()
    
    axs[1, 0].plot(raw_error, label='Raw Error', color='red', linewidth=1.5)
    axs[1, 0].plot(processed_error, label='Processed Error', color='blue', linewidth=1.5)
    axs[1, 0].set_title('Error Analysis', fontsize=16)
    axs[1, 0].set_xlabel('Data Points', fontsize=14)
    axs[1, 0].set_ylabel('Error', fontsize=14)
    axs[1, 0].legend(fontsize=12)
    axs[1, 0].grid(True)
    
    # 1D CNN Prediction vs Actual
    axs[1, 1].scatter(processed_data['X'], processed_data['Y'], c='red', alpha=0.5, label='Actual', s=20)
    axs[1, 1].scatter(predictions[:, 0], predictions[:, 1], c='blue', alpha=0.5, label='1D CNN Prediction', s=20)
    axs[1, 1].set_title('1D CNN Prediction vs Actual', fontsize=16)
    axs[1, 1].set_xlabel('X Coordinate', fontsize=14)
    axs[1, 1].set_ylabel('Y Coordinate', fontsize=14)
    axs[1, 1].legend(fontsize=12)
    axs[1, 1].grid(True)
    
    # 1D CNN Training History
    axs[2, 0].plot(history.history['loss'], label='Training Loss', color='blue', linewidth=1.5)
    axs[2, 0].plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=1.5)
    axs[2, 0].set_title('1D CNN Training History', fontsize=16)
    axs[2, 0].set_xlabel('Epoch', fontsize=14)
    axs[2, 0].set_ylabel('Loss', fontsize=14)
    axs[2, 0].legend(fontsize=12)
    axs[2, 0].grid(True)
    
    # Error Distribution
    sns.histplot(processed_error, kde=True, color='skyblue', edgecolor='black', ax=axs[2, 1])
    axs[2, 1].set_title('Error Distribution', fontsize=16)
    axs[2, 1].set_xlabel('Error', fontsize=14)
    axs[2, 1].set_ylabel('Count', fontsize=14)
    axs[2, 1].grid(True)
    
    # Adjust layout with additional bottom margin
    plt.tight_layout(pad=3.0)  # Increase padding between plots
    plt.subplots_adjust(hspace=0.4, bottom=0.1)  # Increase bottom margin to prevent clipping
    plt.subplots_adjust(top=0.93)  # Adjust top to give space for the main title
    fig.suptitle('Sensor Data Analysis Results', fontsize=24)
    plt.show()

def main():
    raw_data = load_data('collected_data.csv')
    processed_data = augment_data(raw_data)
    processed_data = handle_missing_values(processed_data)
    processed_data = remove_outliers(processed_data)
    processed_data = smooth_data(processed_data)
    
    data_scaled, scaler = standardize_data(processed_data)
    data_pca, pca = apply_pca(data_scaled)
    
    kmeans_labels, gmm_labels, dbscan_labels, agglomerative_labels, spectral_labels = apply_clustering(data_pca)
    ensemble_labels = ensemble_clustering(kmeans_labels, gmm_labels, dbscan_labels, agglomerative_labels, spectral_labels)
    
    evaluate_clustering(data_pca, ensemble_labels)
    
    cnn_features = ['X', 'Y', 'Velocity', 'Acceleration', 'Jerk', 'Rolling_Mean', 'Angle']
    cnn_data = processed_data[cnn_features]
    cnn_target = processed_data[['X', 'Y']]
    
    cnn_model, history, X_test, y_test = train_1d_cnn(cnn_data, cnn_target)
    
    # Reshape X_test for prediction
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    predictions = cnn_model.predict(X_test_reshaped, batch_size=128)
    
    calculate_errors(y_test, predictions)
    
    visualize_results(raw_data, processed_data, predictions, ensemble_labels, history)
    
    raw_mae = mean_absolute_error(raw_data[['X', 'Y']].values[-len(predictions):], predictions)
    processed_mae = mean_absolute_error(y_test, predictions)
    accuracy_improvement = ((raw_mae - processed_mae) / raw_mae) * 100
    
    print(f"Accuracy Improvement: {accuracy_improvement:.2f}%")

if __name__ == "__main__":
    main()
