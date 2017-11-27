from keras.models import model_from_json

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2

def nvidia(input_shape, dropout):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=input_shape))
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Convolution2D(36, (5, 5), strides=(2, 2), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Convolution2D(48, (5, 5), strides=(2, 2), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Convolution2D(64, (3, 3), strides=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Convolution2D(64, (3, 3), strides=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(dropout))
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(dropout))
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dropout(dropout))

    model.add(Dense(1))

    return model


class PilotKeras:
    def run(self, img_arr):
        transformed_image_array = img_arr[None, :, :, :]

        # This model currently assumes that the features of the model are just the images. Feel free to change this.
        steering_angle = float(self.model.predict(transformed_image_array, batch_size=1))
        # The driving model currently just outputs a constant throttle. Feel free to edit this.
        throttle = 0.6
        print(steering_angle, throttle)

        return steering_angle, throttle

    def load(self, model_json_file):
        # with open(model_json_file, 'r') as jfile:
        #     model = model_from_json(jfile.read())
        model = nvidia(input_shape=(120, 160, 3), dropout=0.0)

        model.compile("adam", "mse")

        weights_file = model_json_file.replace('json', 'h5')
        model.load_weights(weights_file)

        self.model_json_file = model_json_file
        self.model = model
