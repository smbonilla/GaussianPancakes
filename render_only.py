import os
from argparse import ArgumentParser
from time import time

parser = ArgumentParser(description="Render only script parameters")
parser.add_argument("--output_path", default="./eval")
args, _ = parser.parse_known_args()

#phantom = ["c3v4", "d4v2", "t1v1"]
phantom = []
#simulation = ["cecum", "rectum", "sigmoid"]
simulation = []
#in_vivo = ["invivo56", "invivo31", "invivo57"] 
in_vivo = []
in_vivo2 = ["051"]

all_scenes = []
if phantom:
    all_scenes.extend(phantom)
if simulation:
    all_scenes.extend(simulation)
if in_vivo: 
    all_scenes.extend(in_vivo)
if in_vivo2:
    all_scenes.extend(in_vivo2)

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
if in_vivo2:
    for scene in in_vivo2:
        all_sources.append("/home/sierra/REIM-NeRF/data3/GP-not-processed/" + scene)

common_args = " --quiet --eval --skip_train --skip_test"
for scene, source in zip(all_scenes, all_sources):
    os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)