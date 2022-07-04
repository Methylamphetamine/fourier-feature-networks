'''
From ''Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains''

https://github.com/tancik/fourier-feature-networks.git

'''

import trimesh
from jax import numpy as np
import os
import numpy as onp

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def recenter_mesh(mesh):
    mesh.vertices -= mesh.vertices.mean(0)
    mesh.vertices /= np.max(np.abs(mesh.vertices))
    mesh.vertices = .5 * (mesh.vertices + 1.)

def uniform_bary(u):
    '''
    calculate the bary center of the triangular mesh
    '''
    su0 = np.sqrt(u[..., 0])
    b0 = 1. - su0
    b1 = u[..., 1] * su0
    return np.stack([b0, b1, 1. - b0 - b1], -1)

def get_normal_batch(mesh, bsize):

    batch_face_inds = np.array(onp.random.randint(0, mesh.faces.shape[0], [bsize]))
    batch_barys = np.array(uniform_bary(onp.random.uniform(size=[bsize, 2])))
    batch_faces = mesh.faces[batch_face_inds]
    batch_normals = mesh.face_normals[batch_face_inds]
    batch_pts = np.sum(mesh.vertices[batch_faces] * batch_barys[...,None], 1)

    return batch_pts, batch_normals

def make_test_pts(mesh, corners, test_size=2**18):
  c0, c1 = corners
  test_easy = onp.random.uniform(size=[test_size, 3]) * (c1-c0) + c0
  batch_pts, batch_normals = get_normal_batch(mesh, test_size)
  test_hard = batch_pts + onp.random.normal(size=[test_size,3]) * .01
  return test_easy, test_hard


def load_mesh(mesh_name, logdir, verbose=True):

    mesh = trimesh.load(mesh_name)
    mesh = as_mesh(mesh)
    if verbose: 
        print(mesh.vertices.shape)
    recenter_mesh(mesh)

    c0, c1 = mesh.vertices.min(0) - 1e-3, mesh.vertices.max(0) + 1e-3
    corners = [c0, c1]
    if verbose:
        print(c0, c1)
        print(c1-c0)
        print(np.prod(c1-c0))
        print(.5 * (c0+c1) * 2 - 1)


    test_pt_file = os.path.join(logdir, mesh_name + '_test_pts.npy')
    if not os.path.exists(test_pt_file):
        if verbose: print('regen pts')
        test_pts = np.array([make_test_pts(mesh, corners), make_test_pts(mesh, corners)])
        np.save(test_pt_file, test_pts)
    else:
        if verbose: print('load pts')
        test_pts = np.load(test_pt_file)

    if verbose: print(test_pts.shape)

    return mesh, corners, test_pts