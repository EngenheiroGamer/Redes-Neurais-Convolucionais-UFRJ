import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam

import os
import matplotlib.pyplot as plt
import datetime
# Define o caminho para o conjunto de dados
data_dir = r'F:/Engenharia/Mestrado/2024.2 Redes Neurais/Códigos/Projeto'

# Assegura que a GPU está disponível
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Carregar o conjunto de dados Places365
dataset, info = tfds.load('places365_small', with_info=True, as_supervised=False, data_dir=data_dir)

def preprocess(image, label):
    # Implementação da pré-processamento que não depende do filename
    # Por exemplo, você pode aplicar transformações na imagem e retornar ela junto com o label
    image = tf.image.resize(image, (224, 224))  # Redimensionar para o tamanho desejado
    image = image / 255.0  # Normalizar os valores
    label = tf.cast(label, tf.int64)  # Certifique-se de que o label seja um inteiro
    label = tf.reshape(label, [])  # Remove dimensões extras do label
    return image, label


# Aplicar a função de preprocessamento
train_ds = dataset['train'].map(lambda x: preprocess(x['image'], x['label'])).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

val_ds = dataset['validation'].map(lambda x: preprocess(x['image'], x['label'])).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

# Definir o modelo com mais camadas
def build_places365_model(input_shape=(224, 224, 3), num_classes=365):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # Nova camada convolucional
        tf.keras.layers.MaxPooling2D((2, 2)),  # Nova camada de pooling
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),  # Nova camada convolucional
        tf.keras.layers.MaxPooling2D((2, 2)),  # Nova camada de pooling
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),  # Nova camada convolucional
        tf.keras.layers.MaxPooling2D((2, 2)),  # Nova camada de pooling
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),  # Com 256 a rede não estava treinando
        tf.keras.layers.Dropout(0.4),  # Aumentei o dropout para prevenir overfitting
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

# Define características do otimizador

    otimizador=Adam(learning_rate = 0.001)
    model.compile(optimizer=otimizador,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'#,
            #tf.keras.metrics.Precision(name='precision'),
            #tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model
# Construir o modelo
model = build_places365_model(num_classes=365)

# Callback para demonstração da CNN
tensorboard_callback =TensorBoard(log_dir = "logs",histogram_freq=1)

# Callback para Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Treinar o modelo com Early Stopping
history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stopping,tensorboard_callback])

# Função para plotar a acurácia e perda
def plot_history(history):
    plt.figure(figsize=(12, 10))

    # Plotar a acurácia
    plt.subplot(3, 1, 1)
    plt.plot(history.history['accuracy'], label='Acurácia Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
    plt.title('Acurácia durante o Treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    # # Plotar a precisão
    # plt.subplot(3, 1, 2)
    # plt.plot(history.history['precision'], label='Precisão Treino')
    # plt.plot(history.history['val_precision'], label='Precisão Validação')
    # plt.title('Precisão durante o Treinamento')
    # plt.xlabel('Épocas')
    # plt.ylabel('Precisão')
    # plt.legend()

    # # Plotar o recall
    # plt.subplot(3, 1, 3)
    # plt.plot(history.history['recall'], label='Recall Treino')
    # plt.plot(history.history['val_recall'], label='Recall Validação')
    # plt.title('Recall durante o Treinamento')
    # plt.xlabel('Épocas')
    # plt.ylabel('Recall')
    # plt.legend()

    # # Plotar a AUC
    # plt.subplot(2, 2, 4)
    # plt.plot(history.history['auc'], label='AUC Treino')
    # plt.plot(history.history['val_auc'], label='AUC Validação')
    # plt.title('AUC durante o Treinamento')
    # plt.xlabel('Épocas')
    # plt.ylabel('AUC')
    # plt.legend()

    plt.show()

# Chamar a função de plotagem
plot_history(history)


# Chamar a função de plotagem
plot_history(history)

# Plota o modelo da CNN
plot_model(model, to_file='Model V1.2.png', show_shapes=True,show_layer_names=True)
