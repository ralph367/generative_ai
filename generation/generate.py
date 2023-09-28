import json
import argparse
import time
import joblib
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras

class Generator:
    def __init__(self, generator_scaler_path, generator_model_path, generator_classes_path, label_distribution_path):
        self.generator_model = load_model(generator_model_path)
        self.generator_scaler = joblib.load(generator_scaler_path)
        self.generator_classes = joblib.load(generator_classes_path)
        self.label_distribution = joblib.load(label_distribution_path)
        self.latent_dim = 128
        self.num_classes = len(self.generator_classes)
        self.freq = 50
    
    def generate_data(self, label):
        gait_time = self.label_distribution[label].rvs(size=1, random_state=random.randint(0, 90))[0]
        gait_len = round(gait_time/(1/self.freq))

        mapped_label = self.generator_classes[label]
        
        generated_data_list = []
    
        for _ in range(gait_len):
            interpolation_noise = tf.random.normal(shape=(1, self.latent_dim))
            interpolation_noise = tf.repeat(interpolation_noise, repeats=1)
            interpolation_noise = tf.reshape(interpolation_noise, (1, 128))
            noise_and_labels = tf.concat([interpolation_noise, keras.utils.to_categorical([mapped_label], self.num_classes)], 1)
            generated_data = self.generator_model.predict(noise_and_labels)
            generated_data = self.generator_scaler.inverse_transform(generated_data.reshape(1, 1))[0][0]
            generated_data_list.append(float(generated_data))
        return generated_data_list, label

class LSTMModel:
    def __init__(self, lstm_model_path, lstm_classes_path):
        self.lstm_model = load_model(lstm_model_path)
        self.lstm_classes = joblib.load(lstm_classes_path)
    
    def predict_label(self, generated_data):
        last_3_game_data = generated_data[-3:] if len(generated_data) >= 3 else []
        if len(last_3_game_data) >= 3:
            label_list = [data["label"] for data in last_3_game_data]
            len_list = [len(data["norm"]) for data in last_3_game_data]
            mean_list = [np.mean(data["norm"]) for data in last_3_game_data]
            range_list = [np.max(data['norm'],  axis=0) - np.min(data['norm'],  axis=0) for data in last_3_game_data]
        else:
            label_list = ["walk", "walk", "walk"]
            len_list = [60, 45, 70]
            mean_list = [40.2, 44.4, 34.2]
            range_list = [23, 45, 32]
        label_class = [self.lstm_classes[x] for x in label_list]
        merged_list = np.stack((np.array([label_class]), np.array([len_list]), np.array([mean_list]), np.array([range_list])), axis=2)
        predictions = self.lstm_model.predict(merged_list)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_label = next((key for key, value in self.lstm_classes.items() if value == predicted_labels[0]), "Walk")
        return predicted_label
    
    
def main():
    parser = argparse.ArgumentParser(description="Generate game data")
    parser.add_argument("--games", type=int, default=1, help="Number of games to generate")
    parser.add_argument("--time", type=int, nargs='+', default=[1], help="List of game times in minutes")
    args = parser.parse_args()

    generator_model_path = 'generator_model.h5'
    generator_scaler_path = 'generator_scaler.pkl'
    generator_classes_path = 'generator_classes.pkl'
    label_distribution_path = 'label_distribution.pkl'
    
    lstm_model_path = 'lstm_model.h5'
    lstm_classes_path = 'lstm_classes.pkl'

    generator = Generator(generator_scaler_path, generator_model_path, generator_classes_path, label_distribution_path)
    lstm_model = LSTMModel(lstm_model_path, lstm_classes_path)

    for game_num in range(1, args.games + 1):
        game_time_minutes = args.time[min(game_num - 1, len(args.time) - 1)]
        game_time_seconds = game_time_minutes * 60

        print(f"Starting Game {game_num} with {game_time_minutes} minutes")
        start_time = time.time()
        
        generated_data = []
        remaining_game_time = game_time_seconds
        initial_label = "walk"
        
        while remaining_game_time > 0:
            generated_data_point, label = generator.generate_data(initial_label)
            game_data = {"label": label, "norm": generated_data_point}
            generated_data.append(game_data)

            predicted_label = lstm_model.predict_label(generated_data)
            
            initial_label = predicted_label
            remaining_game_time -= len(generated_data_point) * 0.02
            print(f"Generated action {label}, {remaining_game_time:.2f} seconds left from game {game_num}")
            
        
        with open(f'generated_data_game{game_num}.json', 'w') as json_file:
            json.dump(generated_data, json_file)
        
        print(f"Game {game_num} completed in: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
