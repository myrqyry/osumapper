# -*- coding: utf-8 -*-

from act_data_prep import *
from act_train_rhythm import *
from gemini_integration import create_multimodal_rhythm_model, load_visual_features

# for osu!mania use this instead of above
# from mania_act_data_prep import *

step1_load_maps();

train_params_p2 = {
    "divisor" : 4,
    "train_epochs" : 16,
    "train_batch_size" : None, # Default is 32 or based on machine specs
    "plot_history" : True,
    "too_many_maps_threshold" : 200,
    "train_epochs_many_maps" : 6,
    "data_split_count" : 80
};

model_p2 = step2_build_model()

# Load visual features
visual_features = []
for i in range(len(os.listdir("mapdata"))):
    visual_fn = os.path.join("mapdata", str(i) + "_visual.npz")
    if os.path.exists(visual_fn):
        visual_features.append(load_visual_features(visual_fn))

# Create the multimodal model
if len(visual_features) > 0:
    visual_features = np.concatenate(visual_features, axis=0)
    model_p2 = create_multimodal_rhythm_model(model_p2, visual_features.shape[1:])

model_p2 = step2_train_model(model_p2, train_params_p2)
step2_evaluate(model_p2)
step2_save(model_p2)
