"""
Run this script to preprocess the mesh for the LA flattening pipeline. It will clean the mesh, fill holes, convert the mesh to a manifold mesh and 
save the cleaned mesh in the Regions folder. It will also provide instructions for the next steps in the pipeline.

Before running this script, make sure you remove any artefacts in the mesh using blender and save that file as <patient>_holes.stl in the Segmentation folder. 
If there are artefacts, the script will skip the filling of large holes and start with the mesh cleaning, converting the mesh into a manifold mesh and 
saving it in the Regions folder.
"""
import os
import sys
import numpy as np
import pyvista as pv
from cleaning_aux_functions import extract_cells_from_non_manifold_edges, remove_problematic_cells_and_save, detect_non_manifold_edges, to_vtk, clean_mesh
from vedo import Mesh, trimesh2vedo
import pyacvd

def to_run(path_to_code, path_to_data, file, eams=False):
    file += ""
    if eams:
        print("Step 1: run" )
        print(f"python {path_to_code}/1_mesh_standardisation.py --meshfile \"{path_to_data}/{file}.vtk\" --pv_dist 3 --laa_dist 3 --vis 1 --save 1 --eams")
        print("Step 2: run" )
        print(f"python {path_to_code}/2_divide_and_flatten_atrium.py --meshfile \"{path_to_data}/{file}_clipped_mitral.vtk\" --save_conts True --save_final_paths True")
    else:
        print("Step 1: run" )
        print(f"python {path_to_code}/1_mesh_standardisation.py --meshfile \"{path_to_data}/{file}.vtk\" --pv_dist 3 --laa_dist 3 --vis 1 --save 1")
        print("Step 2: run" )
        print(f"python {path_to_code}/2_divide_and_flatten_atrium.py --meshfile \"{path_to_data}/{file}_clipped_mitral.vtk\" --save_conts True --save_final_paths True")


if __name__=="__main__":

    path_to_code = os.getcwd()  # Get the current working directory
    path_to_data = ""           # Add path to data here

    for patient in sorted(os.listdir(path_to_data))[1:2]:          #Run for all patients or change index to run for a specific patient
        print(f"Processing patient: {patient}")
        path_to_patient = f"{path_to_data}/{patient}"
        path_to_regions = f"{path_to_patient}/Regions"
        path_to_segmentation = f"{path_to_patient}/Segmentation"
        os.makedirs(path_to_regions, exist_ok=True)

        files = []
        if os.path.exists(f"{path_to_segmentation}/{patient}_holes.stl"):
            files.append(patient+'_holes')
            file = patient+'_holes'

            # Fill holes in the mesh and save it
            mesh = Mesh(f"{path_to_segmentation}/{file}.stl")
            vertices = np.asarray(mesh.vertices())
            faces = np.asarray(mesh.cells())
            vclean, fclean = clean_mesh(vertices, faces) 
            mesh = Mesh([vclean, fclean]) 
            mesh.write(f"{path_to_segmentation}/{patient}_holes_filled.stl", binary=False)
        else:
            for file in os.listdir(path_to_segmentation):
                if not "clean." in file and "AUMC" in file:              # Add constraints here to only process the relevant files, e.g., check for specific substrings in the filename
                    files.append(file[:-4])
                
        for w,file in enumerate(files):
            mesh = pv.PolyData(f"{path_to_segmentation}/{file}.stl")
            if file == patient+'_holes':
                file = patient
                os.remove(f"{path_to_segmentation}/{file}_holes_filled.stl")
            if not os.path.exists(f"{path_to_segmentation}/{file}_clean.vtk"):
                # Remesh the mesh to ensure uniformity
                print(f"Remeshing mesh for patient {patient}, file {file}")
                npoints = len(mesh.points)
                clus = pyacvd.Clustering(mesh)
                clus.subdivide(3)
                clus.cluster(npoints)
                remesh = clus.create_mesh()
                remesh.save(f"{path_to_segmentation}/{file}_remeshed.vtk", binary=False)

                # Clean the mesh
                mesh = Mesh(f"{path_to_segmentation}/{file}_remeshed.vtk")
                vertices = mesh.vertices()
                faces = mesh.cells()
                vclean, fclean = clean_mesh(np.asarray(vertices), np.asarray(faces)) 
                mesh = Mesh([vclean, fclean]) 
                to_vtk(f"{path_to_segmentation}/", file + '_remeshed_clean', mesh)
                
                # Convert to manifold mesh
                problematic_faces, collected_cell_ids = extract_cells_from_non_manifold_edges(f"{path_to_segmentation}/{file}_remeshed_clean.vtk")
                if problematic_faces is not None:
                    problematic_faces.plot()
                remove_problematic_cells_and_save(f"{path_to_segmentation}/{file}_remeshed_clean.vtk", collected_cell_ids, f"{path_to_segmentation}/{file}_clean.vtk", fill_holes=True, hole_size=5)

                n_new_cells = detect_non_manifold_edges(f"{path_to_segmentation}/{file}_clean.vtk")[0]
                if n_new_cells == 0:
                    print("No non-manifold edges detected in the cleaned mesh.")
                else:
                    print(f"Warning: {n_new_cells} non-manifold edges detected in the cleaned mesh. Check if location is not on PVs. Repeating cleaning process.")
                    problematic_faces, collected_cell_ids = extract_cells_from_non_manifold_edges(f"{path_to_segmentation}/{file}_clean.vtk")
                    if problematic_faces is not None:
                        problematic_faces.plot()
                    remove_problematic_cells_and_save(f"{path_to_segmentation}/{file}_clean.vtk", collected_cell_ids, f"{path_to_segmentation}/{file}_clean.vtk", fill_holes=True, hole_size=5)

                # Clean mesh and save it
                vmesh = Mesh(f"{path_to_segmentation}/{file}_clean.vtk")
                vclean, fclean = clean_mesh(np.asarray(vmesh.vertices()), np.asarray(vmesh.cells()))
                mesh = Mesh([vclean, fclean]) 
                to_vtk(path_to_segmentation, file + '_clean', mesh)
                print(f"Cleaned mesh saved to: {path_to_patient}/Segmentation/{file}_clean.vtk")
                to_vtk(path_to_regions, file + '_clean', mesh)
                
                # Remove the intermediate files
                os.remove(path_to_patient + "/Segmentation/" + file + "_remeshed_clean_mesh_wrong_format.vtk")
                os.remove(path_to_patient + "/Segmentation/" + file + "_remeshed.vtk")
                os.remove(path_to_patient + "/Segmentation/" + file + "_remeshed_clean.vtk")
                os.remove(path_to_patient + "/Segmentation/" + file + "_clean_mesh_wrong_format.vtk")
                os.remove(path_to_patient + "/Regions/" + file + "_clean_mesh_wrong_format.vtk")
                
            to_run(path_to_code, path_to_regions, file + "_clean", True)
