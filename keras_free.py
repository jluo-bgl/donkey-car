from keras.models import model_from_json


class PilotKeras:

    def __init__(self, model_json_file):
        self.model_json_file = model_json_file
        self.model = self.load(model_json_file)

    def run(self, img_arr):
        transformed_image_array = img_arr

        # This model currently assumes that the features of the model are just the images. Feel free to change this.
        steering_angle = float(self.model.predict(transformed_image_array, batch_size=1))
        # The driving model currently just outputs a constant throttle. Feel free to edit this.
        throttle = 0.20
        print(steering_angle, throttle)

        return steering_angle, throttle

    @staticmethod
    def load(model_json_file):
        with open(model_json_file, 'r') as jfile:
            model = model_from_json(jfile.read())

        model.compile("adam", "mse")

        weights_file = model_json_file.replace('json', 'h5')
        model.load_weights(weights_file)

        return model
