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


class CowWriter(Writer):
    def __init__(
        self,
        output_dir: str,
        semantic_types: List[str] = None,
        rgb: bool = True,
        bounding_box_2d_tight: bool = False,
        bounding_box_2d_loose: bool = False,
        semantic_segmentation: bool = False,
        instance_id_segmentation: bool = False,
        instance_segmentation: bool = False,
        distance_to_camera: bool = False,
        bounding_box_3d: bool = False,
        image_output_format: str = "png",
    ):
        self._output_dir = output_dir
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._frame_id = 0
        self._sequence_id = 0
        self._image_output_format = image_output_format
        self._output_data_format = {}
        self.annotators = []

        if semantic_types is None:
            semantic_types = ["class"]

        # RGB
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))

        # Bounding Box 2D
        if bounding_box_2d_tight:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("bounding_box_2d_tight", init_params={
                                                "semanticTypes": semantic_types})
            )

        if bounding_box_2d_loose:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("bounding_box_2d_loose", init_params={
                                                "semanticTypes": semantic_types})
            )

        # Semantic Segmentation
        if semantic_segmentation:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "semantic_segmentation",
                    init_params={"semanticTypes": semantic_types},
                )
            )

        # Instance Segmentation
        if instance_id_segmentation:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "instance_id_segmentation", init_params={}
                )
            )

        # Instance Segmentation
        if instance_segmentation:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "instance_segmentation",
                    init_params={"semanticTypes": semantic_types},
                )
            )

        # Depth
        if distance_to_camera:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("distance_to_camera"))

        # Bounding Box 3D
        if bounding_box_3d:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("bounding_box_3d", init_params={
                                                "semanticTypes": semantic_types})
            )

    def write(self, data: dict):

        sequence_id = ""
        for trigger_name, call_count in data["trigger_outputs"].items():
            if "on_time" in trigger_name:
                sequence_id = f"{call_count}_{sequence_id}"
        if sequence_id != self._sequence_id:
            self._frame_id = 0
            self._sequence_id = sequence_id

        for annotator in data.keys():
            annotator_split = annotator.split("-")
            render_product_path = ""
            multi_render_prod = 0
            if len(annotator_split) > 1:
                multi_render_prod = 1
                render_product_name = annotator_split[-1]
                render_product_path = f"{render_product_name}/"

            if annotator.startswith("rgb"):
                if multi_render_prod:
                    render_product_path += "rgb/"
                self._write_rgb(data, render_product_path, annotator)

            if annotator.startswith("distance_to_camera"):
                if multi_render_prod:
                    render_product_path += "distance_to_camera/"
                self._write_distance_to_camera(
                    data, render_product_path, annotator)

            if annotator.startswith("semantic_segmentation"):
                if multi_render_prod:
                    render_product_path += "semantic_segmentation/"
                self._write_semantic_segmentation(
                    data, render_product_path, annotator)

            if annotator.startswith("instance_id_segmentation"):
                if multi_render_prod:
                    render_product_path += "instance_id_segmentation/"
                self._write_instance_id_segmentation(
                    data, render_product_path, annotator)

            if annotator.startswith("instance_segmentation"):
                if multi_render_prod:
                    render_product_path += "instance_segmentation/"
                self._write_instance_segmentation(
                    data, render_product_path, annotator)

            if annotator.startswith("bounding_box_3d"):
                if multi_render_prod:
                    render_product_path += "bounding_box_3d/"
                self._write_bounding_box_data(
                    data, "3d", render_product_path, annotator)

            if annotator.startswith("bounding_box_2d_loose"):
                if multi_render_prod:
                    render_product_path += "bounding_box_2d_loose/"
                self._write_bounding_box_data(
                    data, "2d_loose", render_product_path, annotator)

            if annotator.startswith("bounding_box_2d_tight"):
                if multi_render_prod:
                    render_product_path += "bounding_box_2d_tight/"
                self._write_bounding_box_data(
                    data, "2d_tight", render_product_path, annotator)

        self._frame_id += 1

    def _write_rgb(self, data: dict, render_product_path: str, annotator: str):
        file_path = f"{render_product_path}rgb_{self._sequence_id}{self._frame_id:0}.{self._image_output_format}"
        self._backend.write_image(file_path, data[annotator])

    def _write_distance_to_camera(self, data: dict, render_product_path: str, annotator: str):
        dist_to_cam_data = data[annotator]
        file_path = (
            f"{render_product_path}distance_to_camera_{self._sequence_id}{self._frame_id:0}.npy"
        )
        buf = io.BytesIO()
        np.save(buf, dist_to_cam_data)
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_semantic_segmentation(self, data: dict, render_product_path: str, annotator: str):
        semantic_seg_data = data[annotator]["data"]
        height, width = semantic_seg_data.shape[:2]

        file_path = (
            f"{render_product_path}semantic_segmentation_{self._sequence_id}{self._frame_id:0}.png"
        )
        if self.colorize_semantic_segmentation:
            semantic_seg_data = semantic_seg_data.view(
                np.uint8).reshape(height, width, -1)
            self._backend.write_image(file_path, semantic_seg_data)
        else:
            semantic_seg_data = semantic_seg_data.view(
                np.uint32).reshape(height, width)
            self._backend.write_image(file_path, semantic_seg_data)

        id_to_labels = data[annotator]["info"]["idToLabels"]
        file_path = f"{render_product_path}semantic_segmentation_labels_{self._sequence_id}{self._frame_id:0}.json"
        buf = io.BytesIO()
        buf.write(json.dumps(
            {str(k): v for k, v in id_to_labels.items()}).encode())
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_instance_id_segmentation(self, data: dict, render_product_path: str, annotator: str):
        instance_seg_data = data[annotator]["data"]
        height, width = instance_seg_data.shape[:2]

        file_path = f"{render_product_path}instance_id_segmentation_{self._sequence_id}{self._frame_id:0}.png"
        if self.colorize_instance_id_segmentation:
            instance_seg_data = instance_seg_data.view(
                np.uint8).reshape(height, width, -1)
            self._backend.write_image(file_path, instance_seg_data)
        else:
            instance_seg_data = instance_seg_data.view(
                np.uint32).reshape(height, width)
            self._backend.write_image(file_path, instance_seg_data)

        id_to_labels = data[annotator]["info"]["idToLabels"]
        file_path = f"{render_product_path}instance_id_segmentation_mapping_{self._sequence_id}{self._frame_id:0}.json"
        buf = io.BytesIO()
        buf.write(json.dumps(
            {str(k): v for k, v in id_to_labels.items()}).encode())
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_instance_segmentation(self, data: dict, render_product_path: str, annotator: str):
        instance_seg_data = data[annotator]["data"]
        height, width = instance_seg_data.shape[:2]

        file_path = (
            f"{render_product_path}instance_segmentation_{self._sequence_id}{self._frame_id:0}.png"
        )
        if self.colorize_instance_segmentation:
            instance_seg_data = instance_seg_data.view(
                np.uint8).reshape(height, width, -1)
            self._backend.write_image(file_path, instance_seg_data)
        else:
            instance_seg_data = instance_seg_data.view(
                np.uint32).reshape(height, width)
            self._backend.write_image(file_path, instance_seg_data)

        id_to_labels = data[annotator]["info"]["idToLabels"]
        file_path = f"{render_product_path}instance_segmentation_mapping_{self._sequence_id}{self._frame_id:0}.json"
        buf = io.BytesIO()
        buf.write(json.dumps(
            {str(k): v for k, v in id_to_labels.items()}).encode())
        self._backend.write_blob(file_path, buf.getvalue())

        id_to_semantics = data[annotator]["info"]["idToSemantics"]
        file_path = f"{render_product_path}instance_segmentation_semantics_mapping_{self._sequence_id}{self._frame_id:0}.json"
        buf = io.BytesIO()
        buf.write(json.dumps(
            {str(k): v for k, v in id_to_semantics.items()}).encode())
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_bounding_box_data(self, data: dict, bbox_type: str, render_product_path: str, annotator: str):
        bbox_data_all = data[annotator]["data"]
        print(bbox_data_all)
        id_to_labels = data[annotator]["info"]["idToLabels"]
        prim_paths = data[annotator]["info"]["primPaths"]

        file_path = f"{render_product_path}bounding_box_{bbox_type}_{self._sequence_id}{self._frame_id:0}.npy"
        buf = io.BytesIO()
        np.save(buf, bbox_data_all)
        self._backend.write_blob(file_path, buf.getvalue())

        labels_file_path = f"{render_product_path}bounding_box_{bbox_type}_labels_{self._sequence_id}{self._frame_id:0}.json"
        buf = io.BytesIO()
        buf.write(json.dumps(id_to_labels).encode())
        self._backend.write_blob(labels_file_path, buf.getvalue())

        labels_file_path = f"{render_product_path}bounding_box_{bbox_type}_prim_paths_{self._sequence_id}{self._frame_id:0}.json"
        buf = io.BytesIO()
        buf.write(json.dumps(prim_paths).encode())
        self._backend.write_blob(labels_file_path, buf.getvalue())
        
        target_coco_bbox_data = []
        count = 0
        for bbox_data in bbox_data_all:
           target_bbox_data = {'x_min': bbox_data['x_min'], 'y_min': bbox_data['y_min'],
                               'x_max': bbox_data['x_max'], 'y_max': bbox_data['y_max']}
           width = int(
               abs(target_bbox_data["x_max"] - target_bbox_data["x_min"]))
           height = int(
               abs(target_bbox_data["y_max"] - target_bbox_data["y_min"]))
               
           if width != 2147483647 and height != 2147483647:
#	            filepath = f"rgb_{self._frame_id}.{self._image_output_format}"
#	            self._backend.write_image(filepath, data["rgb"])
               bbox_filepath = f"bbox_{self._frame_id}.txt"
               coco_bbox_data = {
                                 "name": prim_paths[count],
                                 "x_min": int(target_bbox_data["x_min"]),
                                 "y_min": int(target_bbox_data["y_min"]),
                                 "x_max": int(target_bbox_data["x_max"]),
                                 "y_max": int(target_bbox_data["y_max"]),
                                 "width": width,
                                 "height": height}
               target_coco_bbox_data.append(coco_bbox_data)
               count += 1

        buf = io.BytesIO()
        buf.write(json.dumps(target_coco_bbox_data).encode())
        self._backend.write_blob(bbox_filepath, buf.getvalue())


rep.WriterRegistry.register(CowWriter)

camera_positions = [(1720, -1220, 200), (3300, -1220, 200),
                    (3300, -3500, 200), (1720, -3500, 200)]

# Attach Render Product
with rep.new_layer():
    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, (1280, 1280))

# Randomizer Function


def randomize_cows1():
    cows = rep.get.prims(semantics=[('class', 'cow_1')])
    with cows:
        rep.modify.visibility(rep.distribution.choice([True, False]))
    return cows.node

rep.randomizer.register(randomize_cows1)

def randomize_cows2():
    cows = rep.get.prims(semantics=[('class', 'cow_2')])
    with cows:
        rep.modify.visibility(rep.distribution.choice([True, False]))
    return cows.node

rep.randomizer.register(randomize_cows2)

def randomize_cows3():
    cows = rep.get.prims(semantics=[('class', 'cow_3')])
    with cows:
        rep.modify.visibility(rep.distribution.choice([True, False]))
    return cows.node

rep.randomizer.register(randomize_cows3)

def randomize_cows4():
    cows = rep.get.prims(semantics=[('class', 'cow_4')])
    with cows:
        rep.modify.visibility(rep.distribution.choice([True, False]))
    return cows.node

rep.randomizer.register(randomize_cows4)

def randomize_environment():
    envs = ["omniverse://localhost/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr",
                  "omniverse://localhost/NVIDIA/Assets/Skies/Night/moonlit_golf_4k.hdr",
                  "omniverse://localhost/NVIDIA/Assets/Skies/Storm/approaching_storm_4k.hdr"]
    lights = rep.create.light(
       light_type = "Dome",
       position = (2500, -2300, 0),
       intensity = rep.distribution.choice([1., 10., 100., 1000.]),
       texture = rep.distribution.choice(envs)
    )
    return lights.node

rep.randomizer.register(randomize_environment)

# Trigger to call randomizer
with rep.trigger.on_frame(num_frames=10):
    with camera:
        rep.modify.pose(position=rep.distribution.choice(
            camera_positions), look_at=(2500, -2300, 0))
    rep.randomizer.randomize_environment()
    rep.randomizer.randomize_cows1()
    rep.randomizer.randomize_cows2()
    rep.randomizer.randomize_cows3()
    rep.randomizer.randomize_cows4()

writer = rep.WriterRegistry.get('BasicWriter')
writer.initialize(output_dir='C:/Users/anura/Desktop/IndoorRanch_ReplicatorOutputs/NewRun1',
                  rgb=True, bounding_box_2d_tight=True)
writer.attach([render_product])











































