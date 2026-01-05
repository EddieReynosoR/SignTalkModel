import numpy as np
import os
from pathlib import Path
from keras.src.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout
from keras.src.callbacks import ModelCheckpoint
from keras.src.regularizers import L2

from sklearn.model_selection import train_test_split 
from keras.src.utils import to_categorical

base_dir = Path(__file__).resolve().parent
DATA_PATH = Path(os.path.join(base_dir, 'MP_Data'))

sequence_length = 30

words = np.array([f for f in os.listdir(DATA_PATH) if (DATA_PATH/f).is_dir()])

if len(words) < 2:
    raise ValueError("Se necesitan al menos 2 palabras para entrenar de forma correcta.")

words.sort()

label_map = {label:num for num, label in enumerate(words)}

sequences, labels = [], []
for word in words:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, word))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, word, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[word])

x = np.array(sequences)
y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

reg = L2(0.01)

# === Definición del modelo ===
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh',
               input_shape=(30, 258),
               kernel_regularizer=reg, recurrent_regularizer=reg))
model.add(Dropout(0.2))

model.add(LSTM(128, return_sequences=True, activation='tanh',
               kernel_regularizer=reg, recurrent_regularizer=reg))
model.add(Dropout(0.2))

model.add(LSTM(64, return_sequences=False, activation='tanh',
               kernel_regularizer=reg, recurrent_regularizer=reg))
model.add(Dropout(0.2))

model.add(Dense(64, activation='tanh', kernel_regularizer=reg))
model.add(Dropout(0.2))

model.add(Dense(32, activation='tanh', kernel_regularizer=reg))
model.add(Dense(words.shape[0], activation='softmax'))


cp_best_val_loss = ModelCheckpoint(
      "SLD_val_loss.weights.h5", monitor='val_loss', mode = 'min', save_weights_only=True, save_best_only=True, verbose=1
)
cp_best_val_acc = ModelCheckpoint(
      "SLD_val_acc.weights.h5", monitor='val_categorical_accuracy', mode = 'max', save_weights_only=True, save_best_only=True, verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# === Reduce LR on Plateau ===
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

# === Entrenamiento ===
history = model.fit(
    x_train, y_train,
    epochs=2000,
    batch_size=8,
    validation_data=(x_test, y_test),
    callbacks=[cp_best_val_loss, cp_best_val_acc, early_stop, reduce_lr, tb_callback],
    verbose=1
)

model.save("sign_model.keras")
model.export("sign_model")