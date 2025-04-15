import numpy as np
import open3d as o3d
import random
import torch
import sys
from param import parse_args
import models
import MinkowskiEngine as ME
from utils.data import normalize_pc
from utils.misc import load_config
from huggingface_hub import hf_hub_download
from collections import OrderedDict
import open_clip
import re
from PIL import Image
import torch.nn.functional as F
import glob
import os
import pymeshlab

def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists. If it doesn't, create it.

    Parameters:
    - directory_path (str): The path to the directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        
def convert_glb_to_ply(glb_path : str,
                       ply_path : str, 
                       least_vertex_num = 10000):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(glb_path)
    
    if ms.mesh_number() > 1:
        ms.generate_by_merging_visible_meshes(mergevisible=False)
        
    assert ms.mesh_number() == 1
    
    while ms.current_mesh().vertex_number() < least_vertex_num:
        ms.apply_filter("meshing_surface_subdivision_catmull_clark")
        
    if not ms.current_mesh().has_vertex_color() and ms.current_mesh().texture_number() > 0:
        ms.transfer_texture_to_color_per_vertex()
    
    ms.save_current_mesh(ply_path, save_vertex_color=True, binary=False,
                         save_textures=False, save_wedge_texcoord=False, save_vertex_coord=False, 
                         save_vertex_quality=False, save_vertex_radius=False, save_face_quality=False,
                         save_wedge_color=False, save_wedge_normal=False)
        
def convert_glb_to_ply_mix(glb_path : str,
                           ply_path : str, 
                           least_vertex_num = 10000):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(glb_path)
    
    if ms.mesh_number() == 1:
        # increase vertex number if necessary
        while ms.current_mesh().vertex_number() < least_vertex_num:
            ms.apply_filter("meshing_surface_subdivision_catmull_clark")

        # transfer texture colors to vertex
        if not ms.current_mesh().has_vertex_color() and ms.current_mesh().texture_number() > 0:
            ms.transfer_texture_to_color_per_vertex()
            
        ms.save_current_mesh(ply_path, save_vertex_color=True, binary=False,
                             save_textures=False, save_wedge_texcoord=False, save_vertex_coord=False, 
                             save_vertex_quality=False, save_vertex_radius=False, save_face_quality=False,
                             save_wedge_color=False, save_wedge_normal=False)
    else:
        # o3d loads the glb as one entire mesh    
        model = o3d.io.read_triangle_model(glb_path)

        mesh = model.meshes[0].mesh
    
        vertex_num = np.array(mesh.vertices).shape[0]
    
        vertex_colors = np.array(mesh.vertex_colors)
    
        if vertex_colors.size == 0:
            # transfer texture colors to vertex
            texture = model.materials[0].albedo_img
            uvs = mesh.triangle_uvs
            if texture != None and uvs != None:
                texture_image = np.asarray(texture)
                
                vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.float32)

                for i, uv in enumerate(uvs):
                    # Get the corresponding vertex index
                    vertex_index = mesh.triangles[i // 3][i % 3]
                    
                    # Sample the texture at the UV coordinate
                    u, v = uv
                    x = int(u * (texture_image.shape[1] - 1))
                    y = int((1 - v) * (texture_image.shape[0] - 1))  # Flip V coordinate
                    color = texture_image[y, x] / 255.0  # Normalize to [0, 1]
                    
                    # Assign the color to the vertex
                    vertex_colors[vertex_index] = color

                # Assign the colors to the mesh
                mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
        o3d.io.write_triangle_mesh(ply_path, mesh, write_ascii=True, compressed=False, 
                                   write_vertex_normals=False,
                                   write_vertex_colors=(vertex_colors.size > 0),    
                                   write_triangle_uvs=False)
    
        if vertex_num < least_vertex_num:
            # use meshlab to subdivide the mesh
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(ply_path)
            
            while ms.current_mesh().vertex_number() < least_vertex_num:
                ms.apply_filter("meshing_surface_subdivision_catmull_clark")
                
            ms.save_current_mesh(ply_path, save_vertex_color=(vertex_colors.size > 0), binary=False,
                                 save_textures=False, save_wedge_texcoord=False, save_vertex_coord=False, 
                                 save_vertex_quality=False, save_vertex_radius=False, save_face_quality=False,
                                 save_wedge_color=False, save_wedge_normal=False)

def load_ply(file_name, num_points=10000, y_up=True):
    pcd = o3d.io.read_point_cloud(file_name)
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    
    print(xyz.shape)
    print(rgb.shape)

    if rgb is None or rgb.size != xyz.size:
        rgb = np.ones_like(xyz) * 0.4

    n = xyz.shape[0]
    if n != num_points:
        idx = random.sample(range(n), num_points)
        xyz = xyz[idx]
        rgb = rgb[idx]
    if y_up:
        # swap y and z axis
        xyz[:, [1, 2]] = xyz[:, [2, 1]]
    xyz = normalize_pc(xyz)
    #if rgb is None:
    #    rgb = np.ones_like(xyz) * 0.4
    features = np.concatenate([xyz, rgb], axis=1)
    xyz = torch.from_numpy(xyz).type(torch.float32)
    features = torch.from_numpy(features).type(torch.float32)
    return ME.utils.batched_coordinates([xyz], dtype=torch.float32), features

def load_model(config, model_name="OpenShape/openshape-spconv-all"):
    model = models.make(config).cuda()

    if config.model.name.startswith('Mink'):
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model) # minkowski only
    else:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in checkpoint['state_dict'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
    model.load_state_dict(model_dict)
    return model

# for k, mp in enumerate(model_ply_paths):
#     print(k)
#     xyz, feat = load_ply(mp)
#     shape_feat = model(xyz, feat, device='cuda', quantization_size=config.model.voxel_size) 
#     np.save(os.path.join(dst_dir, os.path.basename(mp).split('.')[0] + ".npy"), shape_feat.detach().cpu().numpy().squeeze())

def init_openshape_model():
    print("loading OpenShape model...")
    cli_args, extras = parse_args([])
    config = load_config("src/configs/train.yaml", cli_args = vars(cli_args), extra_args = extras)
    model = load_model(config)
    model.eval()
    
    return model, config

def main():
    glb_path = "./demo/man.glb"
    ply_path = glb_path + ".ply"
    
    convert_glb_to_ply(glb_path, ply_path)
    
    model, config = init_openshape_model()
    
    xyz, feat = load_ply(ply_path)
    
    shape_feat = model(xyz, feat, device='cuda', quantization_size=config.model.voxel_size) 
    
    np.save("shape_feat.npy", shape_feat.detach().cpu().numpy().squeeze())

    print("Done!")

if __name__ == "__main__":
    main()
