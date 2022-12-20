import io
import json
import time
import asyncio
from typing import List

import omni.kit
import omni.usd
import omni.replicator.core as rep

import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry, orchestrator
from omni.syntheticdata.scripts.SyntheticData import SyntheticData

camera_positions = [(1720, -1220, 200), (3300, -1220, 200),
                    (3300, -3500, 200), (1720, -3500, 200)]

# Attach Render Product
with rep.new_layer():
    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, (1280, 1280))

# Randomizer Function


def randomize_cows():
    cows = rep.get.prims(semantics=[('class', 'cow')])
    with cows:
        rep.modify.visibility(rep.distribution.choice([True, False]))
    return cows.node


rep.randomizer.register(randomize_cows)

# Trigger to call randomizer
with rep.trigger.on_frame(num_frames=10):
    with camera:
        rep.modify.pose(position=rep.distribution.choice(
            camera_positions), look_at=(2500, -2300, 0))
    rep.randomizer.randomize_cows()

# Initialize and attach Writer to store result
writer = rep.WriterRegistry.get('CowWriter')
writer.initialize(output_dir='C:/Users/anura/Desktop/IndoorRanch_ReplicatorOutputs/Run4',
                  rgb=True, bounding_box_2d_tight=True)
writer.attach([render_product])











