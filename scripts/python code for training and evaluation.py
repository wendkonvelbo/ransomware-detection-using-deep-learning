import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

# --- 1. Dataset Loading and Preprocessing ---
def load_and_preprocess_data(filepath):
    print("Loading and preprocessing the dataset...")
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found. Please ensure the path and filename are correct.")
        return None, None

    data.columns = data.columns.str.strip()
    X = data.drop(['FileName', 'Benign'], axis=1)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    y = data['Benign'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Dataset loaded and scaled successfully.")
    print(f"Number of samples: {len(data)}")
    print(f"Number of features: {X.shape[1]}")

    return X_scaled, y

# --- 2. Standalone Models (10 Epochs) ---

def run_standalone_cnn(X, y):
    print("\n--- Training Standalone CNN Model ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    class_weights_dict = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        Conv1D(128, 3, activation='relu', padding='same', input_shape=(X_train_reshaped.shape[1], 1)),
        MaxPooling1D(pool_size=2), Dropout(0.3),
        Conv1D(256, 3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2), Dropout(0.3),
        Conv1D(512, 3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2), Dropout(0.3),
        Flatten(), Dense(128, activation='relu'), Dropout(0.5), Dense(1, activation='sigmoid')
    ])
    
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=128, validation_split=0.2, class_weight=class_weights_dict, verbose=1)
    y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int)
    print("\nStandalone CNN Performance:")
    print(classification_report(y_test, y_pred))
    plot_results(history, confusion_matrix(y_test, y_pred), "Standalone CNN")

def run_standalone_lstm(X, y):
    print("\n--- Training Standalone LSTM Model ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    class_weights_dict = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(X_train_reshaped.shape[1], 1)),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=128, validation_split=0.2, class_weight=class_weights_dict, verbose=1)
    y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int)
    print("\nStandalone LSTM Performance:")
    print(classification_report(y_test, y_pred))
    plot_results(history, confusion_matrix(y_test, y_pred), "Standalone LSTM")

def run_standalone_rnn(X, y):
    print("\n--- Training Standalone RNN Model ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    class_weights_dict = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        SimpleRNN(128, return_sequences=True, input_shape=(X_train_reshaped.shape[1], 1)),
        Dropout(0.3),
        SimpleRNN(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=128, validation_split=0.2, class_weight=class_weights_dict, verbose=1)
    y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int)
    print("\nStandalone RNN Performance:")
    print(classification_report(y_test, y_pred))
    plot_results(history, confusion_matrix(y_test, y_pred), "Standalone RNN")

def run_standalone_dnn(X, y):
    print("\n--- Training Standalone DNN Model ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    class_weights_dict = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2, class_weight=class_weights_dict, verbose=1)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("\nStandalone DNN Performance:")
    print(classification_report(y_test, y_pred))
    plot_results(history, confusion_matrix(y_test, y_pred), "Standalone DNN")

# --- 3. Hybrid Models (50 Epochs) ---

def run_hybrid_cnn_lstm(X, y):
    print("\n--- Training Hybrid CNN-LSTM Model ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    class_weights_dict = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        Conv1D(64, 3, activation='relu', padding='same', input_shape=(X_train_reshaped.shape[1], 1)),
        MaxPooling1D(pool_size=2), Dropout(0.3),
        Conv1D(128, 3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2), Dropout(0.3),
        Conv1D(256, 3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2), Dropout(0.3),
        LSTM(128),
        Dense(64, activation='relu'), Dropout(0.5), Dense(1, activation='sigmoid')
    ])
    
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=128, validation_split=0.2, class_weight=class_weights_dict, verbose=1)
    y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int)
    print("\nHybrid CNN-LSTM Performance:")
    report = classification_report(y_test, y_pred)
    print(report)
    plot_results(history, confusion_matrix(y_test, y_pred), "Hybrid CNN-LSTM")
    print_hybrid_precision(report)
# --- 2. Combined RNN and DNN Model ---
def run_combined_rnn_dnn(X, y):
    print("\n--- Training Combined RNN and DNN Model ---")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    
    # Reshape for RNN: (samples, timesteps=features, channels=1)
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        SimpleRNN(128, return_sequences=False, input_shape=(X_train_reshaped.shape[1], 1)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'), # Added a second Dense layer
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=50, # Increased epochs to 50
        batch_size=128,
        validation_split=0.2,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    y_pred_proba = model.predict(X_test_reshaped)
    y_pred = (y_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nCombined RNN-DNN Performance:")
    print(classification_report(y_test, y_pred))
    plot_results(history, cm, "Combined RNN-DNN")

# --- 3. Plotting Functions ---
def plot_results(history, cm, model_name):
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'Model Accuracy ({model_name})')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'Model Loss ({model_name})')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    plot_confusion_matrix(cm, model_name)

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malware', 'Benign'],
                yticklabels=['Malware', 'Benign'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    filepath = r'D:\ADMIN\AAAA\data_file.csv'
    
    X, y = load_and_preprocess_data(filepath)
    if X is None:
        exit()

    run_combined_rnn_dnn(X, y)
# --- 3. Plotting Functions ---
def plot_results(history, cm, model_name):
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'Model Accuracy ({model_name})')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'Model Loss ({model_name})')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    plot_confusion_matrix(cm, model_name)

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malware', 'Benign'],
                yticklabels=['Malware', 'Benign'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# --- 5. Custom Function to Print Precision ---
def print_hybrid_precision(report):
    lines = report.split('\n')
    precision_line = [line for line in lines if 'precision' in line]
    if precision_line:
        print("\n--- Hybrid Model Precision ---")
        print(precision_line[0])
    else:
        print("Precision data not found in report.")

# --- 6. Plotting and Main Execution ---
def plot_results(history, cm, model_name):
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'Model Accuracy ({model_name})')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'Model Loss ({model_name})')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    plot_confusion_matrix(cm, model_name)

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malware', 'Benign'],
                yticklabels=['Malware', 'Benign'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    filepath = r'D:\ADMIN\AAAA\data_file.csv'
    X, y = load_and_preprocess_data(filepath)
    if X is None:
        exit()

    run_standalone_cnn(X, y)
    run_standalone_lstm(X, y)
    run_standalone_dnn(X, y)
    run_standalone_rnn(X, y)
    run_hybrid_cnn_lstm(X, y)
    run_hybrid_dnn_rnn(X, y)