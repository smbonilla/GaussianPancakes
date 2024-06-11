#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser
from time import time

# phantom = ["c3v4", "d4v2", "t1v1"]
phantom = []
# simulation = ["cecum", "rectum", "sigmoid"]
simulation = []

in_vivo = []
#in_vivo = ["invivo56", "invivo57", "invivo31"]
in_vivo = ["invivo56", "invivo31", "invivo57"] 

in_vivo_2 = ["039", "055", "063", "035", "067"]
#in_vivo_2 = []

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
args, _ = parser.parse_known_args()

all_scenes = []
if phantom:
    all_scenes.extend(phantom)
if simulation:
    all_scenes.extend(simulation)
if in_vivo: 
    all_scenes.extend(in_vivo)
if in_vivo_2:
    all_scenes.extend(in_vivo_2)

all_times = []

if not args.skip_training:
    #common_args = " --quiet --eval --test_iterations -1 "
    common_args = " --quiet --eval --test_iterations -1 "
    if phantom:
        for scene in phantom:
            start = time()
            source = "/home/sierra/data/01-phantom/" + scene
            os.system("python train.py -s " + source + " --iterations 7000 --res 4 -m " + args.output_path + "/" + scene + common_args)
            end = time()
            print("Training time for " + scene + ": " + str(end - start) + " seconds")
            all_times.append(end - start)
    if simulation:
        for scene in simulation:
            start = time()
            source = "/home/sierra/REIM-NeRF/data/GP-not-processed/" + scene
            os.system("python train.py -s " + source + " --iterations 7000 -m " + args.output_path + "/" + scene + common_args)
            end = time()
            print("Training time for " + scene + ": " + str(end - start) + " seconds")
            all_times.append(end - start)
    if in_vivo:
        for scene in in_vivo:
            start = time()
            source = "/home/sierra/data/00-in-vivo/" + scene
            os.system("python train.py -s " + source + " --iterations 7000 -m " + args.output_path + "/" + scene + common_args)
            end = time()
            print("Training time for " + scene + ": " + str(end - start) + " seconds")
            all_times.append(end - start)
    if in_vivo_2:
        for scene in in_vivo_2:
            start = time()
            source = "/home/sierra/REIM-NeRF/data3/GP-not-processed/" + scene
            os.system("python train.py -s " + source + " --iterations 7000 -m " + args.output_path + "/" + scene + common_args)
            end = time()
            print("Training time for " + scene + ": " + str(end - start) + " seconds")
            all_times.append(end - start)

if not args.skip_rendering:
    all_sources = []
    if phantom:
        for scene in phantom:
            all_sources.append("/home/sierra/data/01-phantom/" + scene)
    if simulation:
        for scene in simulation:
            all_sources.append("/home/sierra/REIM-NeRF/data/GP-not-processed/" + scene)
    if in_vivo:
        for scene in in_vivo:
            all_sources.append("/home/sierra/data/00-in-vivo/" + scene)
    if in_vivo_2:
        for scene in in_vivo_2:
            all_sources.append("/home/sierra/REIM-NeRF/data3/GP-not-processed/" + scene)

    common_args = " --quiet --eval"
    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    if all_times:
        os.system("python metrics.py -m " + scenes_string + " -t " + " ".join([str(t) for t in all_times]))
    else:
        os.system("python metrics.py -m " + scenes_string)