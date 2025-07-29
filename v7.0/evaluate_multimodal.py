# -*- coding: utf-8 -*-

from act_newmap_prep import *
from act_rhythm_calc import *
from act_gan import *;
from act_modding import *
from act_final import *
from gemini_integration import create_multimodal_rhythm_model, load_visual_features

# input file here! (don't remove the "r" before string)
file_path = r'..\\..\\test_data\\test.osu'

# Or use auto timing with music file only!!

# from act_timing import *;
# music_path = r\"..\\..\\test_data\\audio.mp3"
# file_path = get_timed_osu_file(music_path, game_mode=0);

step4_read_new_map(file_path);

model = step5_load_model();
npz, visual_fn = step5_load_npz();

visual_features = None
if os.path.exists(visual_fn):
    visual_features = load_visual_features(visual_fn)

if visual_features is not None:
    model = create_multimodal_rhythm_model(model, visual_features.shape[1:])

params = step5_set_params(dist_multiplier=1, note_density=0.35, slider_favor=0, divisor_favor=[0] * 4, slider_max_ticks=8);

predictions = step5_predict_notes(model, (npz, visual_fn), params);
converted = step5_convert_sliders(predictions, params);

step5_save_predictions(converted);

gan_params = {
    "divisor" : 4,
    "good_epoch" : 12,
    "max_epoch" : 30,
    "note_group_size" : 10,
    "g_epochs" : 1,
    "c_epochs" : 1,
    "g_batch" : 50,
    "g_input_size" : 50,
    "c_true_batch" : 140,
    "c_false_batch" : 5,
    "c_randfalse_batch" : 5,
    "note_distance_basis" : 200,
    "next_from_slider_end" : False,
    "max_ticks_for_ds" : 1,
    "box_loss_border" : 0.1,
    "box_loss_value" : 0.4,
    "box_loss_weight" : 1
};

step6_set_gan_params(gan_params);
osu_a, data = step6_run_all();

modding_params = {
    "stream_regularizer" : 1,
    "slider_mirror" : 1
}

osu_a, data = step7_modding(osu_a, data, modding_params);

saved_osu_name = step8_save_osu_file(osu_a, data);

# for taiko mode only (comment out the above line and use below)
# from act_taiko_hitsounds import *
# taiko_hitsounds_params = step8_taiko_hitsounds_set_params(divisor=4, metronome_count=4)
# hitsounds = step8_apply_taiko_hitsounds(osu_a, data, params=taiko_hitsounds_params)
# saved_osu_name = step8_save_osu_file(osu_a, data, hitsounds=hitsounds);

# clean up the folder
step8_clean_up();
