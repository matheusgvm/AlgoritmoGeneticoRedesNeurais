from tensorflow import keras
import pandas as pd
from keras.callbacks import EarlyStopping


class CriarRedeNeural:
    def __init__(self,  data_path):
        self.data_path = data_path
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.preparar_dataset()
        self.num_execucoes = 0
        self.cache_results = {}

    def executar(self, num_camadas_ocultas, num_neuronios_por_camada_oculta, batch_size):

        key = str(num_camadas_ocultas)
        for num in num_neuronios_por_camada_oculta:
            key += str(num)
        key += str(batch_size)

        if key in self.cache_results:
            return self.cache_results.get(key)

        self.num_execucoes += 1

        early_stopping = EarlyStopping(
            min_delta=0.001,  # minimium amount of change to count as an improvement
            patience=10,  # how many epochs to wait before stopping
            restore_best_weights=True,
        )

        model = keras.models.Sequential()

        for i in range(num_camadas_ocultas):
            model.add(keras.layers.Dense(units=num_neuronios_por_camada_oculta[i], activation='relu', input_shape=[self.X_train.shape[1]]))

        model.add(keras.layers.Dense(1))
        model.compile(
            optimizer='adam',
            loss='mse'
        )

        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_valid, self.y_valid),
            batch_size=batch_size,
            epochs=500,
            callbacks=[early_stopping]
        )

        ultimo_val_loss = round(history.history['val_loss'][-1], 3)
        # Subtrai a paciencia
        melhor_epoca = early_stopping.stopped_epoch - 10
        self.cache_results[key] = [ultimo_val_loss, melhor_epoca]

        return [ultimo_val_loss, melhor_epoca]
        #return -ultimo_val_loss, early_stopping.stopped_epoch

    def preparar_dataset(self):
        red_wine = pd.read_csv(self.data_path)

        # Create training and validation splits
        df_train = red_wine.sample(frac=0.7, random_state=0)
        df_valid = red_wine.drop(df_train.index)

        # Scale to [0, 1]
        max_ = df_train.max(axis=0)
        min_ = df_train.min(axis=0)
        df_train = (df_train - min_) / (max_ - min_)
        df_valid = (df_valid - min_) / (max_ - min_)

        # Split features and target
        self.X_train = df_train.drop('quality', axis=1)
        self.X_valid = df_valid.drop('quality', axis=1)
        self.y_train = df_train['quality']
        self.y_valid = df_valid['quality']


# if __name__ == '__main__':
#     rede = CriarRedeNeural('winequality-red.csv')
#     a, b = rede.executar(5, [3,3,3,3,3], 32)
#     print(a)
#     print(b)

