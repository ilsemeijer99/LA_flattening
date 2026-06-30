"""
    Copyright (c) - Marta Nunez Garcia
    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option)
    any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
    Public License for more details. You should have received a copy of the GNU General Public License along with this
    program. If not, see <http://www.gnu.org/licenses/>.
"""

"""
    Implementation of left atrial (LA) flattening in "Fast quasi-conformal regional flattening of the left atrium", Nunez-Garcia, Marta, et al., 2018, arXiv preprint arXiv:1811.06896

    Input: LA mesh with clipped & filled holes (PVs, LAA) and only 1 hole corresponding to MV.
    Output: Flat (2D) version of input mesh.
    Usage: python 4_flat_atria.py --meshfile data/mesh_clipped_c.vtk

    Conformal flattening considering 6 boundaries (4 PVs + LAA + MV) and additional regional constraints
    Regional constraints fitted using segments: s1,s2,s3,s4,s5,s6,s7, s8a and s8b
    Boundaries fitted using segments: s9, s10, s11, s12, rspv_s{1,2,3}, ripv_s{1,2,3}, lipv_s{1,2,3}, lspv_s{1,2,3}, laa_s{1,2}
    Constraints (division lines) are modified to enforce proportional distribution of points in the holes.
"""
import vtk
from aux_functions import *
from clip_aux_functions import vis_paths
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex

import sys
import os
import argparse
from cleaning_aux_functions import *
parser = argparse.ArgumentParser()
parser.add_argument('--meshfile', type=str, metavar='PATH', help='path to input mesh')
parser.add_argument('--save_conts', type=bool, default=False, help='set to true to save mesh contours/contraints')
parser.add_argument('--save_final_paths', type=bool, default=False, help='set to true to save modified dividing paths')
args = parser.parse_args()
fileroot = os.path.dirname(args.meshfile)
filename = os.path.basename(args.meshfile)
filenameroot = os.path.splitext(filename)[0][:-15]
if os.path.isfile(args.meshfile)==False:
    sys.exit('ERROR: input file does not exist')
else:
    m_open = readvtk(args.meshfile)

to_be_flat_path = os.path.join(fileroot, filenameroot + '_to_be_flat.vtk')
output_path = os.path.join(fileroot, filenameroot + '_flat.vtk')
input_path = os.path.join(fileroot, filenameroot + '_autolabels.vtp')

m_whole = readvtp(input_path)
##################   Template creation. Define position and radius of PV holes, and radius disk. Adapt to input mesh.    ##################
rdisk = 0.5
pv_centers = np.zeros([2, 4])
r_min = 0.03
rhole_lipv = r_min
rhole_lspv = 1.1*r_min
rhole_ripv = 1.1*r_min
rhole_rspv = 1.35*r_min
rhole_laa = 1.35*r_min
laa_disp_x = 0.03  # displacement of LAA wrt LSPV (in x direction, to the left)

px_ref = -0.25
py_ref = -0.10
left_carina_length = 0.175
right_carina_length = 1.1 * left_carina_length   # real proportions, do not separate more, it induces distortion close to the holes
pwall_width = 2.6 * left_carina_length
sep_lspv_laa = 1.2
t_v5_1 = np.pi / 8
t_v5_2 = 1* np.pi / 3
t_v6 = 2*np.pi - np.pi/6
t_v7 = np.pi + np.pi / 6
t_v8 = 3 * np.pi / 4 - np.pi/40

# rspv
pv_centers[0, 0] = px_ref + pwall_width
pv_centers[1, 0] = py_ref + left_carina_length + (right_carina_length - left_carina_length) / 2
# ripv
pv_centers[0, 1] = px_ref + pwall_width
pv_centers[1, 1] = py_ref - (right_carina_length - left_carina_length) / 2
# lipv
pv_centers[0, 2] = px_ref
pv_centers[1, 2] = py_ref
# lspv
pv_centers[0, 3] = px_ref
pv_centers[1, 3] = py_ref + left_carina_length
# LAA
laa_hole_center_x = px_ref - laa_disp_x
laa_hole_center_y = py_ref + left_carina_length + left_carina_length * sep_lspv_laa  # si lipv esta en (-.25, -.10)

xhole_center = pv_centers[0, :]
yhole_center = pv_centers[1, :]

# define the proportion of points in each of the PV segments
alpha = np.arctan(np.divide(laa_disp_x, laa_hole_center_y - pv_centers[1, 3]))  # angle of the line connecting LSPV and LAA
proportions = define_pv_segments_proportions(t_v5_1, t_v5_2, t_v6, t_v7, alpha)
propn_rspv_s1, propn_rspv_s2, propn_rspv_s3, propn_rspv_s4 = proportions[0, :]
propn_ripv_s1, propn_ripv_s2, propn_ripv_s3, _ = proportions[1, :]
propn_lipv_s1, propn_lipv_s2, propn_lipv_s3, _ = proportions[2, :]
propn_lspv_s1, propn_lspv_s2, propn_lspv_s3, _ = proportions[3, :]
propn_laa_s1, propn_laa_s2, propn_laa_s3, _ = proportions[4, :]
propn_mv_s1, propn_mv_s2,  propn_mv_s3, propn_mv_s4 = define_mv_segments_proportions()

##################    Open PVs and LAA holes (get 'to_be_flat_mesh'), identify contours and dividing paths in the to_be_flat mesh   ##################
# contours
m_whole, cont_rspv, cont_ripv, cont_lipv, cont_lspv, cont_mv, cont_laa = extract_LA_contours(m_open, args.meshfile, m_whole,  args.save_conts)

locator, locator_open, locator_rspv, locator_ripv, locator_lipv, locator_lspv, locator_laa = build_locators(m_whole, m_open, cont_rspv, cont_ripv, cont_lipv, cont_lspv, cont_laa)
mv_cont_ids = get_mv_contour_ids(cont_mv, locator_open).astype(int)
lipv_cont_ids = get_mv_contour_ids(cont_lipv, locator_open).astype(int)
lspv_cont_ids = get_mv_contour_ids(cont_lspv, locator_open).astype(int)
ripv_cont_ids = get_mv_contour_ids(cont_ripv, locator_open).astype(int)
rspv_cont_ids = get_mv_contour_ids(cont_rspv, locator_open).astype(int)
laa_cont_ids = get_mv_contour_ids(cont_laa, locator_open).astype(int)

path1 = find_create_path_contours(m_open, rspv_cont_ids, ripv_cont_ids)
path2 = find_create_path_contours(m_open, ripv_cont_ids, lipv_cont_ids)
path3 = find_create_path_contours(m_open, lipv_cont_ids, lspv_cont_ids)
path4 = find_create_path_contours(m_open, lspv_cont_ids, rspv_cont_ids)
path5 = find_create_path_contours(m_open, mv_cont_ids, rspv_cont_ids)
path6 = find_create_path_contours(m_open, mv_cont_ids, ripv_cont_ids)
path7 = find_create_path_contours(m_open, mv_cont_ids, lipv_cont_ids)
path8a = find_create_path_contours(m_open, laa_cont_ids, lspv_cont_ids)
path8b = find_create_path_contours(m_open, mv_cont_ids, laa_cont_ids)
path8c = find_create_path_contours(m_open, laa_cont_ids, rspv_cont_ids)    

# Check for overlap between path IDs and contour IDs
path1, path2, path3, path4, path5, path6, path7, path8a, path8b, path8c = check_and_adjust_paths(locator_open, path1, path2, path3, path4, path5, path6, path7, path8a, path8b, path8c,
                                                                            ripv_cont_ids, rspv_cont_ids, lipv_cont_ids, lspv_cont_ids, laa_cont_ids)

writevtk(path1, os.path.join(fileroot, filenameroot + '_path1.vtk'))
writevtk(path2, os.path.join(fileroot, filenameroot + '_path2.vtk'))
writevtk(path3, os.path.join(fileroot, filenameroot + '_path3.vtk'))
writevtk(path4, os.path.join(fileroot, filenameroot + '_path4.vtk'))
writevtk(path5, os.path.join(fileroot, filenameroot + '_path5.vtk'))
writevtk(path6, os.path.join(fileroot, filenameroot + '_path6.vtk'))
writevtk(path7, os.path.join(fileroot, filenameroot + '_path7.vtk'))
writevtk(path8a, os.path.join(fileroot, filenameroot + '_path8a.vtk'))
writevtk(path8b, os.path.join(fileroot, filenameroot + '_path8b.vtk'))
writevtk(path8c, os.path.join(fileroot, filenameroot + '_path8c.vtk'))

# find point on intersection mv and path5,6,7,8
id_v5 = np.intersect1d(mv_cont_ids,get_ids(path5, locator_open)).astype(int)[0]
id_v6 = np.intersect1d(mv_cont_ids,get_ids(path6, locator_open)).astype(int)[0]
id_v7 = np.intersect1d(mv_cont_ids,get_ids(path7, locator_open)).astype(int)[0]
id_v8 = np.intersect1d(mv_cont_ids,get_ids(path8b, locator_open)).astype(int)[0]

# define target coordinates in the disk and get them separated
coordinates = define_disk_template(rdisk, rhole_rspv, rhole_ripv, rhole_lipv, rhole_lspv, rhole_laa, xhole_center, yhole_center,laa_hole_center_x, laa_hole_center_y, t_v5_1, t_v5_2, t_v6, t_v7, t_v8)
v1r_x, v1r_y, v1d_x, v1d_y, v1l_x, v1l_y, v2u_x, v2u_y, v2r_x, v2r_y, v2l_x, v2l_y, v3u_x, v3u_y, v3r_x, v3r_y, v3l_x, v3l_y, v4r_x, v4r_y, v4u_x, v4u_y, v4d_x, v4d_y, vlaad_x, vlaad_y, vlaau_x, vlaau_y, p5_x, p5_y, p6_x, p6_y, p7_x, p7_y, p8_x, p8_y, v1u_x, v1u_y, vlaar_x, vlaar_y = get_coords(coordinates)

# Obtain ids corresponding to the extremes of the segments
v1r, v1d, v1l, v1u, v2u, v2r, v2l, v3u, v3r, v3l, v4r, v4u, v4d, vlaad, vlaau, vlaar = identify_segments_extremes(path1, path2, path3, path4, path5, path6, path7, path8a, path8b, path8c,
                               locator_open, locator_rspv, locator_ripv, locator_lipv, locator_lspv, locator_laa, cont_rspv, cont_ripv, cont_lipv, cont_lspv, cont_laa)

# Get ids of the segments. Start with PV holes. Start with angle = 0 or pi (anticlock-wise, positive angle direction)
rspv_ids, rspv_s1_prop, rspv_s2_prop, rspv_s3_prop, rspv_s4_prop, v1l_prop, v1d_prop, v1r_prop, v1u_prop = get_rspv_segments_ids(cont_rspv, locator_open, v1l, v1d, v1r, v1u, propn_rspv_s1, propn_rspv_s2, propn_rspv_s3, propn_rspv_s4)
ripv_ids, ripv_s1_prop, ripv_s2_prop, ripv_s3_prop, v2l_prop, v2r_prop, v2u_prop = get_ripv_segments_ids(cont_ripv, locator_open, v2l, v2r, v2u, propn_ripv_s1, propn_ripv_s2, propn_ripv_s3)
lipv_ids, lipv_s1_prop, lipv_s2_prop, lipv_s3_prop, v3r_prop, v3u_prop, v3l_prop = get_lipv_segments_ids(cont_lipv, locator_open, v3r, v3u, v3l, propn_lipv_s1, propn_lipv_s2, propn_lipv_s3)
lspv_ids, lspv_s1_prop, lspv_s2_prop, lspv_s3_prop, v4r_prop, v4u_prop, v4d_prop = get_lspv_segments_ids(cont_lspv, locator_open, v4r, v4u, v4d, propn_lspv_s1, propn_lspv_s2, propn_lspv_s3)
laa_ids, laa_s1, laa_s2, laa_s3, vlaau_prop, vlaad_prop, vlaar_prop = get_laa_segments_ids(cont_laa, locator_open, vlaau, vlaad, vlaar, propn_laa_s1, propn_laa_s2, propn_laa_s3)
mv_ids, mv_s1_prop, mv_s2_prop, mv_s3_prop, mv_s4_prop, v5_prop, v6_prop, v7_prop, v8_prop = get_mv_segments_ids(cont_mv, locator_open, id_v5, id_v6, id_v7, id_v8, propn_mv_s1, propn_mv_s2, propn_mv_s3, propn_mv_s4)

# write updated paths after displacing segment extremes to have proportional PV segments lengths
path1_clipped_prop, v1d_prop, v2u_prop = find_trimmed_path_between_contours(m_open, int(v1d_prop), int(v2u_prop), rspv_ids, ripv_ids)  # Always second to first to get order ids order from first to second
path2_clipped_prop, v2l_prop, v3r_prop = find_trimmed_path_between_contours(m_open, int(v2l_prop), int(v3r_prop), ripv_ids, lipv_ids)
path3_clipped_prop, v3u_prop, v4d_prop = find_trimmed_path_between_contours(m_open, int(v3u_prop), int(v4d_prop), lipv_ids, lspv_ids)
path4_clipped_prop, v4r_prop, v1l_prop = find_trimmed_path_between_contours(m_open, int(v4r_prop), int(v1l_prop), lspv_ids, rspv_ids)
path5_clipped_prop, v1r_prop, v5_prop = find_trimmed_path_between_contours(m_open, int(v1r_prop), int(v5_prop), rspv_ids, mv_ids)
path6_clipped_prop, v2r_prop, v6_prop = find_trimmed_path_between_contours(m_open, int(v2r_prop), int(v6_prop), ripv_ids, mv_ids)
path7_clipped_prop, v3l_prop, v7_prop = find_trimmed_path_between_contours(m_open, int(v3l_prop), int(v7_prop), lipv_ids, mv_ids) 
path8a_clipped_prop, v4u_prop, vlaad_prop = find_trimmed_path_between_contours(m_open, int(v4u_prop), int(vlaad_prop), lspv_ids, laa_ids)
path8b_clipped_prop, vlaau_prop, v8_prop = find_trimmed_path_between_contours(m_open, int(vlaau_prop), int(v8_prop), laa_ids, mv_ids)
path8c_clipped_prop, v1u_prop, vlaar_prop = find_trimmed_path_between_contours(m_open, int(v1u_prop), int(vlaar_prop), rspv_ids, laa_ids)

rspv_ids = np.append(rspv_ids[int(np.where(rspv_ids==v1d_prop)[0]):len(rspv_ids)], rspv_ids[0:int(np.where(rspv_ids==v1d_prop)[0])])
rspv_s1_prop, rspv_s2_prop, rspv_s3_prop, rspv_s4_prop = rspv_ids[0:int(np.where(rspv_ids==v1l_prop)[0])],rspv_ids[int(np.where(rspv_ids==v1l_prop)[0]):int(np.where(rspv_ids==v1u_prop)[0])],rspv_ids[int(np.where(rspv_ids==v1u_prop)[0]):int(np.where(rspv_ids==v1r_prop)[0])],rspv_ids[int(np.where(rspv_ids==v1r_prop)[0]):len(rspv_ids)] 
pos_v2l = int(np.where(ripv_ids == v2l_prop)[0])
ripv_ids = np.append(ripv_ids[pos_v2l:ripv_ids.size], ripv_ids[0:pos_v2l])
ripv_s1_prop = ripv_ids[0:int(np.where(ripv_ids == v2r_prop)[0])]
ripv_s2_prop = ripv_ids[int(np.where(ripv_ids == v2r_prop)[0]): int(np.where(ripv_ids == v2u_prop)[0])]
ripv_s3_prop = ripv_ids[int(np.where(ripv_ids == v2u_prop)[0]): ripv_ids.size]
pos_v3u = int(np.where(lipv_ids == v3u_prop)[0])
lipv_ids = np.append(lipv_ids[pos_v3u:lipv_ids.size], lipv_ids[0:pos_v3u])
lipv_s1_prop = lipv_ids[0:int(np.where(lipv_ids == v3l_prop)[0])]
lipv_s2_prop = lipv_ids[int(np.where(lipv_ids == v3l_prop)[0]): int(np.where(lipv_ids == v3r_prop)[0])]
lipv_s3_prop = lipv_ids[int(np.where(lipv_ids == v3r_prop)[0]): lipv_ids.size]
pos_v4r = int(np.where(lspv_ids == v4r_prop)[0])
lspv_ids = np.append(lspv_ids[pos_v4r:lspv_ids.size], lspv_ids[0:pos_v4r])
lspv_s1_prop = lspv_ids[0:int(np.where(lspv_ids == v4u_prop)[0])]
lspv_s2_prop = lspv_ids[int(np.where(lspv_ids == v4u_prop)[0]): int(np.where(lspv_ids == v4d_prop)[0])]
lspv_s3_prop = lspv_ids[int(np.where(lspv_ids == v4d_prop)[0]): lspv_ids.size]
pos_vlaar = int(np.where(laa_ids == vlaar_prop)[0]) 
laa_ids = np.append(laa_ids[pos_vlaar:laa_ids.size], laa_ids[0:pos_vlaar])
laa_s1 = laa_ids[0:int(np.where(laa_ids == vlaau_prop)[0])]
laa_s2 = laa_ids[int(np.where(laa_ids == vlaau_prop)[0]): int(np.where(laa_ids == vlaad_prop)[0])]
laa_s3 = laa_ids[int(np.where(laa_ids == vlaad_prop)[0]): laa_ids.size]

if args.save_final_paths == True:
    writevtk(path1_clipped_prop, os.path.join(fileroot, filenameroot + '_path1_prop.vtk'))
    writevtk(path2_clipped_prop, os.path.join(fileroot, filenameroot + '_path2_prop.vtk'))
    writevtk(path3_clipped_prop, os.path.join(fileroot, filenameroot + '_path3_prop.vtk'))
    writevtk(path4_clipped_prop, os.path.join(fileroot, filenameroot + '_path4_prop.vtk'))
    writevtk(path5_clipped_prop, os.path.join(fileroot, filenameroot + '_path5_prop.vtk'))
    writevtk(path6_clipped_prop, os.path.join(fileroot, filenameroot + '_path6_prop.vtk'))
    writevtk(path7_clipped_prop, os.path.join(fileroot, filenameroot + '_path7_prop.vtk'))
    writevtk(path8a_clipped_prop, os.path.join(fileroot, filenameroot + '_path8a_prop.vtk'))
    writevtk(path8b_clipped_prop, os.path.join(fileroot, filenameroot + '_path8b_prop.vtk'))
    writevtk(path8c_clipped_prop, os.path.join(fileroot, filenameroot + '_path8c_prop.vtk'))

##################    Define segment constraint points (s1, s2, s3,..., s12)    ##################
s1 = get_segment_ids_in_to_be_flat_mesh(path1_clipped_prop, locator_open, np.concatenate([ripv_s2_prop, ripv_s3_prop]), np.concatenate([rspv_s1_prop, rspv_s4_prop]))
s2 = get_segment_ids_in_to_be_flat_mesh(path2_clipped_prop, locator_open, np.concatenate([lipv_s2_prop, lipv_s3_prop]), np.concatenate([ripv_s1_prop, ripv_s3_prop]))
s3 = get_segment_ids_in_to_be_flat_mesh(path3_clipped_prop, locator_open, np.concatenate([lspv_s2_prop, lspv_s3_prop]), np.concatenate([lipv_s1_prop, lipv_s3_prop]))
s4 = get_segment_ids_in_to_be_flat_mesh(path4_clipped_prop, locator_open, np.concatenate([rspv_s1_prop, rspv_s2_prop]), np.concatenate([lspv_s1_prop, lspv_s3_prop]))
s5 = get_segment_ids_in_to_be_flat_mesh(path5_clipped_prop, locator_open, mv_cont_ids, np.concatenate([rspv_s3_prop, rspv_s4_prop]))
s6 = get_segment_ids_in_to_be_flat_mesh(path6_clipped_prop, locator_open, mv_cont_ids, np.concatenate([ripv_s1_prop, ripv_s2_prop]))
s7 = get_segment_ids_in_to_be_flat_mesh(path7_clipped_prop, locator_open, mv_cont_ids, np.concatenate([lipv_s1_prop, lipv_s2_prop]))
s8a = get_segment_ids_in_to_be_flat_mesh(path8a_clipped_prop, locator_open, np.concatenate([laa_s2, laa_s3]), np.concatenate([lspv_s1_prop, lspv_s2_prop]))
s8b = get_segment_ids_in_to_be_flat_mesh(path8b_clipped_prop, locator_open, mv_cont_ids, np.concatenate([laa_s1, laa_s2]))
s8c = get_segment_ids_in_to_be_flat_mesh(path8c_clipped_prop, locator_open, np.concatenate([laa_s1, laa_s3]), np.concatenate([rspv_s2_prop, rspv_s3_prop]))

# Concatenate all constraint segments: s1,s2,s3,s4,s5,s6,s7, s8a and s8b
auxx = np.append(s1, s2)
auxx = np.append(auxx, s3)
auxx = np.append(auxx, s4)
auxx = np.append(auxx, s5)
auxx = np.append(auxx, s6)
auxx = np.append(auxx, s7)
auxx = np.append(auxx, s8a)
auxx = np.append(auxx, s8b)
seq_constraints_ids = np.append(auxx, s8c).astype(int)

##################    Define contour constraint points    ##################
# Concatenate ids of the contour segments: 3 in each veins (PV hole) + 4 external disk + 2 for LAA
auxx2 = np.append(mv_s1_prop, mv_s2_prop)
auxx2 = np.append(auxx2, mv_s3_prop)
auxx2 = np.append(auxx2, mv_s4_prop)
auxx2 = np.append(auxx2, rspv_s1_prop)
auxx2 = np.append(auxx2, rspv_s2_prop)
auxx2 = np.append(auxx2, rspv_s3_prop)
auxx2 = np.append(auxx2, rspv_s4_prop)
auxx2 = np.append(auxx2, ripv_s1_prop)
auxx2 = np.append(auxx2, ripv_s2_prop)
auxx2 = np.append(auxx2, ripv_s3_prop)
auxx2 = np.append(auxx2, lipv_s1_prop)
auxx2 = np.append(auxx2, lipv_s2_prop)
auxx2 = np.append(auxx2, lipv_s3_prop)
auxx2 = np.append(auxx2, lspv_s1_prop)
auxx2 = np.append(auxx2, lspv_s2_prop)
auxx2 = np.append(auxx2, lspv_s3_prop)
auxx2 = np.append(auxx2, laa_s1)
auxx2 = np.append(auxx2, laa_s2)
seq_contour_ids = np.append(auxx2, laa_s3).astype(int)

# check repeated points -> singular matrix and no solution
counts1 = np.bincount(seq_constraints_ids)
counts2 = np.bincount(seq_contour_ids)
counts3 = np.bincount(np.concatenate([seq_constraints_ids, seq_contour_ids]))
print('Number of repeated constraints', np.where(counts1 > 1)[0])
print('Number of repeated contour conditions', np.where(counts2 > 1)[0])
print('Number of repeated constraints and conditions', np.where(counts3 > 1)[0])

find_location_of_repeated_ids(np.where(counts3>1)[0], s1, s2, s3, s4, s5, s6, s7, s8a, s8b, s8c, mv_s1_prop, mv_s2_prop, mv_s3_prop, mv_s4_prop, 
                            rspv_s1_prop, rspv_s2_prop, rspv_s3_prop, rspv_s4_prop, ripv_s1_prop, ripv_s2_prop, ripv_s3_prop, lipv_s1_prop, lipv_s2_prop, lipv_s3_prop, 
                            lspv_s1_prop, lspv_s2_prop, lspv_s3_prop, laa_s1, laa_s2, laa_s3)

# put together all PV and LAA segment sizes
segment_lengths = np.zeros([6, 4])
segment_lengths[0, 0] = rspv_s1_prop.size
segment_lengths[0, 1] = rspv_s2_prop.size
segment_lengths[0, 2] = rspv_s3_prop.size
segment_lengths[0, 3] = rspv_s4_prop.size
segment_lengths[1, 0] = ripv_s1_prop.size
segment_lengths[1, 1] = ripv_s2_prop.size
segment_lengths[1, 2] = ripv_s3_prop.size
segment_lengths[2, 0] = lipv_s1_prop.size
segment_lengths[2, 1] = lipv_s2_prop.size
segment_lengths[2, 2] = lipv_s3_prop.size
segment_lengths[3, 0] = lspv_s1_prop.size
segment_lengths[3, 1] = lspv_s2_prop.size
segment_lengths[3, 2] = lspv_s3_prop.size
segment_lengths[4, 0] = laa_s1.size
segment_lengths[4, 1] = laa_s2.size
segment_lengths[4, 2] = laa_s3.size
segment_lengths[5, 0] = mv_s1_prop.size
segment_lengths[5, 1] = mv_s2_prop.size
segment_lengths[5, 2] = mv_s3_prop.size
segment_lengths[5, 3] = mv_s4_prop.size

##################    Create target (x0, y0) positions  according to the lengths of each segment    ##################
# Separately constraints and contours
x0_const, y0_const = define_constraints_positions(s1, s2, s3, s4, s5, s6, s7, s8a, s8b, s8c, v1l_x, v1l_y, v1d_x, v1d_y, v1r_x, v1r_y, v1u_x, v1u_y, v2l_x,
                                 v2l_y, v2r_x, v2r_y, v2u_x, v2u_y, v3r_x, v3r_y, v3u_x, v3u_y, v3l_x, v3l_y,
                                 v4r_x, v4r_y, v4u_x, v4u_y, v4d_x, v4d_y, vlaad_x, vlaad_y, vlaau_x, vlaau_y,vlaar_x, vlaar_y, p5_x,
                                 p5_y, p6_x, p6_y, p7_x, p7_y, p8_x, p8_y)

x0_bound, y0_bound = define_boundary_positions(rdisk, rhole_rspv, rhole_ripv, rhole_lipv, rhole_lspv, rhole_laa, xhole_center, yhole_center, laa_hole_center_x, laa_hole_center_y, segment_lengths.astype(int), t_v5_1, t_v5_2, t_v6, t_v7, t_v8, args)
plt.plot(x0_bound, y0_bound, 'ro')
plt.plot(x0_const, y0_const, 'bx')
plt.title('2D template with boundary (red) and regional (blue) constraint points')
plt.show()

f = open(args.meshfile[0:len(args.meshfile)-19] + '_flat_lines.txt', 'w')
for point_x0_const, point_y0_const in zip(x0_const, y0_const):
    f.write(str(point_x0_const) + " ")
    f.write(str(point_y0_const) + " ")
    f.write("\n")
f.close()

m_flat = flat_w_constraints(m_open, seq_contour_ids.astype(int), seq_constraints_ids.astype(int), x0_bound, y0_bound, x0_const, y0_const)
m_final = flat(m_flat, seq_contour_ids.astype(int), x0_bound, y0_bound)   # Refine boundary

# Add region (R1, R2, R3, R4, R5) label to the _to_be_flat mesh and the final flat mesh
# summarize and write all dividing lines in a txt file
line_textfile = args.meshfile[0:len(args.meshfile)-19] + '_div_lines.txt'
nlines = 10

locator = vtk.vtkPointLocator()
locator.SetDataSet(m_open)
locator.BuildLocator()
f = open(line_textfile, 'w')

for i in range(1, nlines+1):
    if i == 1:
        path = path1_clipped_prop
    elif i == 2:
        path = path2_clipped_prop
    elif i == 3:
        path = path3_clipped_prop
    elif i == 4:
        path = path4_clipped_prop
    elif i == 5:
        path = path5_clipped_prop
    elif i == 6:
        path = path6_clipped_prop
    elif i == 7:
        path = path7_clipped_prop
    elif i == 8:
        path = path8a_clipped_prop
    elif i == 9:
        path = path8b_clipped_prop
    elif i == 10:
        path = path8c_clipped_prop
    ids = np.arange(0, path.GetNumberOfPoints())
    for p in range(path.GetNumberOfPoints()):
        id_p = locator.FindClosestPoint(path.GetPoint(ids[p]))
        f.write(str(id_p))
        f.write(' ')
    f.write('\n')
f.close()

# Read all lines and divide/cut mesh
m_aux = set_piece_label_from_contours(m_open, rspv_cont_ids, ripv_cont_ids, lspv_cont_ids, lipv_cont_ids, laa_cont_ids, mv_cont_ids, line_textfile, to_be_flat_path)
writevtk(m_aux, to_be_flat_path)

print('\nProjecting information...')
transfer_all_scalar_arrays_by_point_id(m_open, m_final)
f = open(os.path.join(fileroot, filenameroot +"_regions.txt"), "w")
for reg in vtk_to_numpy(m_aux.GetCellData().GetArray('region')):
    f.write(str(reg))
    f.write('\n')
f.close()
m_final.GetCellData().AddArray(m_aux.GetCellData().GetArray('region'))
# remove ad hoc scalar arrays
#m_final.GetPointData().RemoveArray('pv')
#m_final.GetPointData().RemoveArray('autolabels')
m_final.GetPointData().RemoveArray('hole')
print('\nRemoving ad hoc scalar arrays: autolabels, pv, and hole')
writevtk(m_final, output_path)

array_labels = np.zeros(m_whole.GetNumberOfCells())
locator = vtk.vtkPointLocator()
locator.SetDataSet(m_whole)
locator.BuildLocator()
locator2 = vtk.vtkCellLocator()
locator2.SetDataSet(m_aux)
locator2.BuildLocator()
for c in range(m_whole.GetNumberOfCells()):
    cell = np.array([list(m_whole.GetPoint(m_whole.GetCell(c).GetPointIds().GetId(i))) for i in range(3)])
    point = np.array([np.mean(cell[:,0]), np.mean(cell[:,1]), np.mean(cell[:,2])])    
    cell_id = locator2.FindCell(point)
    if cell_id == -1:
        point_id = locator.FindClosestPoint(point)
        autolabel = m_whole.GetPointData().GetArray("autolabels").GetValue(point_id)
        array_labels[c] = autolabel
    else:
        
        region = m_aux.GetCellData().GetArray("region").GetValue(cell_id)
        array_labels[c] = region
newarray = numpy_to_vtk(array_labels)
newarray.SetName("region")
m_whole.GetCellData().AddArray(newarray)
writevtk(m_whole, os.path.join(fileroot, filenameroot + "_regions.vtk"))


region_remap = {36: 12, 37: 7, 76: 11, 77: 10, 78: 9, 79: 8}

regions = array_labels
print(np.unique(regions))
regions = np.array([region_remap.get(r, r) for r in regions], dtype=int)
print(np.unique(regions))
region_dict = {1:"Septal", 2:"Inferior", 3:"Lateral",  4:"Roof", 5:"Posterior", 6:"Anterior",  7:"LAA", 8:"LIPV", 9:"LSPV", 10:"RIPV", 11:"RSPV", 12:'Mitral Valve' }


plot_regions(m_whole, regions, region_dict)

# from vedo import Plotter, build_lut, load
# import matplotlib
# vmesh = load(os.path.join(fileroot, filenameroot + "_regions.vtk"))
# plot = Plotter(offscreen=False)
# lut_table = []
# cmap_regions1 = matplotlib.cm.get_cmap('tab20')

# for i, regionname in enumerate(np.unique(regions)):    
#     lut_table.append((regionname, to_hex(cmap_regions1.colors[i]), 1, region_dict[regionname]))  
# print(lut_table)
# lut = build_lut(lut_table)
# title = "Regions"
# vmesh.cmap(lut, regions.astype(int), on="cells")
# vmesh.add_scalarbar3d(title="Regions", categories=lut_table)
# plot.add(vmesh)
# plot.show()