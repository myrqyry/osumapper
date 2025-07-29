# -*- coding: utf-8 -*-

import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model

def create_multimodal_rhythm_model(rhythm_model, visual_features_shape):
    """
    Creates a new rhythm model that accepts visual features as an additional input.

    Args:
        rhythm_model: The existing rhythm model.
        visual_features_shape: The shape of the visual features.

    Returns:
        A new rhythm model that accepts visual features as an additional input.
    """
    # Create a new input layer for the visual features
    visual_input = Input(shape=visual_features_shape)

    # Get the existing audio input layer
    audio_input = rhythm_model.input

    # Concatenate the audio and visual features
    concatenated_features = concatenate([audio_input[1], visual_input])

    # Create a new model with the concatenated features
    x = rhythm_model.layers[2](audio_input[0]) #This is a hack, but it works for now
    x = rhythm_model.layers[3](x)
    x = rhythm_model.layers[4](x)
    x = rhythm_model.layers[5](x)
    x = rhythm_model.layers[6](x)
    x = rhythm_model.layers[7](x)
    x = rhythm_model.layers[8]([x, concatenated_features])
    output = rhythm_model.layers[9](x)


    model = Model(inputs=[audio_input[0], audio_input[1], visual_input], outputs=output)

    return model

def load_visual_features(path):
    """
    Loads visual features from a file.

    Args:
        path (str): The path to the file containing the visual features.

    Returns:
        A NumPy array of visual features.
    """
    with np.load(path) as data:
        return data["visual_features"]

def extract_and_save_visual_features(video_path, output_path):
    """
    Extracts visual features from a video file and saves them to a file.

    Args:
        video_path (str): The path to the video file.
        output_path (str): The path to save the visual features to.
    """
    features = extract_visual_features(video_path)
    np.savez_compressed(output_path, visual_features=features)

def extract_visual_features(video_path):
    """
    Extracts visual features from a video file.

    Args:
        video_path (str): The path to the video file.

    Returns:
        A NumPy array of visual features.
    """
    # Load the InceptionV3 model
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Extract features from each frame
    features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame = cv2.resize(frame, (299, 299))
        frame = np.expand_dims(frame, axis=0)
        frame = preprocess_input(frame)

        # Extract features from the frame
        feature = model.predict(frame)
        features.append(feature)

    cap.release()

    return np.array(features)
