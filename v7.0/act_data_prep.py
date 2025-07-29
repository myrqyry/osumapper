# -*- coding: utf-8 -*-

#
# Part 1 action script
#

from audio_tools import *;
from os_tools import *;
from gemini_integration import extract_and_save_visual_features;

import os, re, time;

mapdata_path = "mapdata/";
try:
    divisor = GLOBAL["divisor"];
except:
    divisor = 4;

def step1_load_maps():
    # fix the path..?
    fix_path()

    # Test paths and node
    test_process_path("node");
    if not os.path.isdir(mapdata_path):
        os.mkdir(mapdata_path);

    # Test node modules..?
    test_node_modules()

    # Test ffmpeg..?
    test_process_path("ffmpeg", "-version");

    # Open maplist
    with open("maplist.txt", encoding="utf8") as fp:
        fcont = fp.readlines();

    # Reset results
    results = [];
    for line in fcont:
        results.append(line);

    # Remove maps
    for file in os.listdir(mapdata_path):
        if file.endswith(".npz"):
            os.remove(os.path.join(mapdata_path, file));

    print("Number of filtered maps: {}".format(len(results)));

    for k, mname in enumerate(results):
        try:
            start = time.time()
            read_and_save_osu_file(mname.strip(), filename=os.path.join(mapdata_path, str(k)), divisor=divisor);
            video_path = mname.strip().replace(".osu", ".mp4")
            if os.path.exists(video_path):
                extract_and_save_visual_features(video_path, os.path.join(mapdata_path, str(k) + "_visual.npz"))
            end = time.time()
            print("Map data #" + str(k) + " saved! time = " + str(end - start) + " secs");
        except Exception as e:
            print("Error on #{}, path = {}, error = {}".format(str(k), mname.strip(), e));