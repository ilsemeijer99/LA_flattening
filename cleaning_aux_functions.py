import pyvista as pv 
import os
import numpy as np
import argparse
import sys
from aux_functions import *
from vedo import write
import re
import pymeshfix
"""
There are conflicting version issues with vmtk, vtk, and pyvista. Mesh processing such as cleaning degenerate and non manifold edges is not embedded 
in pure vtk. Pyvista is a wrapper around vtk and has more advanced mesh processing capabilities. 
Other mesh processing libvraries could have been used such as trimesh of pyMeshlab. However, pyvista is more user friendly and does not require additionnal mesh
conversion as it wraps around vtk. 

TTK is another interesting option to explore but it's way more complex and requires a lot of dependencies.
"""

class Converter:
    def __init__(self, inp, out=None, copy_lines=False):
        if out is None:
            out = inp
        self.inp = inp
        self.out = out
        self.copy_lines = copy_lines  # if line cells should be copied
        self.original = None
        self.lines = None
        self.polys = None
        self.cells = None

    def read(self):
        with open(self.inp, "r") as f:
            self.original = f.read().split("\n")
        lines_original = list(map(lambda x: x.strip().split(), self.original))

        for i, l in enumerate(self.original):
            if "LINES" in l:
                self.lines = NewContent("LINES", lines_original, i)
            elif "POLYGONS" in l:
                self.polys = NewContent("POLYGONS", lines_original, i)
            elif "CELLS" in l:
                self.cells = NewContent("CELLS", lines_original, i)

    def replace(self):
        if self.polys is not None:
            self.original = self.polys.replace(self.original)
        if self.cells is not None:
            self.original = self.cells.replace(self.original)
        if self.lines is not None:
            self.original = self.lines.replace(self.original, replace=self.copy_lines)

    def write(self):
        with open(self.out, "w") as f:
            f.write("\n".join(self.original))

        change_vtk_version(self.out, 4.2)


class NewContent:
    def __init__(self, kw, content, ln):
        self.kw = kw
        self.ln = ln
        self.name = content[ln][0]
        self.no = int(content[ln][1])
        self.nc = int(content[ln][2])

        flat_list = [item for line in content[ln + 2 :] for item in line]
        flat_list = list(filter("".__ne__, flat_list))

        self.offsets = list(map(int, flat_list[0 : self.no]))
        self.connectivity = list(
            map(int, flat_list[self.no + 2 : self.no + 2 + self.nc])
        )

    @property
    def remove(self):
        return self.ln, self.ln + math.ceil(self.no / 9) + math.ceil(self.nc / 9) + 3

    def replace(self, lines, replace=True):
        nb_cells = self.no - 1
        new_content = []
        if replace:
            new_content = [f"{self.kw} {nb_cells} {nb_cells + self.nc}"]
            for i in range(nb_cells):
                nb_points = self.offsets[i + 1] - self.offsets[i]
                ids = self.connectivity[self.offsets[i] : self.offsets[i + 1]]
                new_content.append(f"{nb_points} {' '.join(map(str, ids))}")

        lines_to_keep = lines
        a, b = self.remove
        del lines_to_keep[a:b]
        lines_to_keep[a:a] = new_content
        lines_to_keep = list(filter("".__ne__, lines_to_keep))
        return lines_to_keep

def convert_to_flattening_vtk_version(mesh_fname: str, save_fname: str, copy_lines: bool = False):
    conv = Converter(mesh_fname, save_fname, copy_lines)
    conv.read()
    conv.replace()
    conv.write()

def change_vtk_version(filename: str, v: float = 5.1):
    with open(filename, "r") as f:
        x = f.read()
    x = re.sub(r"(# vtk DataFile Version) (.+)", f"\\1 {v}", x)
    with open(filename, "w") as f:
        f.write(x)

def to_vtk(main, file, mesh):   
    write(mesh,main + "\\"+ file +  "_mesh_wrong_format.vtk", binary=False)
    convert_to_flattening_vtk_version(main+ "\\" + file +  "_mesh_wrong_format.vtk", main+ "\\" +  file + ".vtk")

def clean_mesh(vertices, faces):
    vclean, fclean = pymeshfix.clean_from_arrays(vertices, faces)
    return vclean, fclean

def remove_non_manifold_pyvista(input_filepath, output_filepath):
    """
    Detect and remove non-manifold edges from a mesh using PyVista.

    Parameters:
    - input_filepath: path to the input mesh file (.vtk or .ply)
    - output_filepath: path to save the cleaned mesh (.vtk or .ply)

    Returns:
    - cleaned pyvista.PolyData object
    """

    mesh = pv.read(input_filepath)
    cleaned_mesh = mesh.clean()
    cleaned_mesh.save(output_filepath)

    return cleaned_mesh


def detect_non_manifold_edges(input_filepath):
    """
    Detect non-manifold edges in a mesh using PyVista.

    Parameters:
    - input_filepath: path to the input mesh file (.vtk or .ply)

    Returns:
    - number of non-manifold edges
    - pyvista.PolyData of the non-manifold edges
    """
    mesh = pv.read(input_filepath)

    # Use PyVista's feature_edges filter
    non_manifold_edges = mesh.extract_feature_edges(
        boundary_edges=False,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=True
    )

    print(f"Detected {non_manifold_edges.n_cells} non-manifold edges.")

    return non_manifold_edges.n_cells, non_manifold_edges


def map_edge_points_to_original_mesh(edge_line, original_mesh):

    """
    Input : edge_line : pyvista.PolyData of the edge line
            original_mesh : pyvista.PolyData of the original mesh   
    Output : point_ids : list of point IDs in the original mesh that correspond to the edge line points
    Description :       
    Map the points of an edge line to the original mesh points.
    """    

    # Get coordinates of the edge points
    edge_points = edge_line.points

    # Get coordinates of the original mesh points
    mesh_points = original_mesh.points

    point_ids = []
    for pt in edge_points:
        # Find closest point index in original mesh
        distances = np.linalg.norm(mesh_points - pt, axis=1)
        closest_id = np.argmin(distances)
        point_ids.append(closest_id)

    return point_ids


def extract_cells_from_non_manifold_edges(input_filepath):
    """
    Detect non-manifold edges and extract the connected faces (triangles),
    using PyVista's neighbor search.

    Parameters:
    - input_filepath: path to the input mesh file (.vtk or .ply)

    Returns:
    - pyvista.PolyData of the selected triangles (cells)
    """
    mesh = pv.read(input_filepath)

    # Detect non-manifold edges
    non_manifold_edges = mesh.extract_feature_edges(
        boundary_edges=False,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=True
    )

    if non_manifold_edges.n_cells == 0:
        print("No non-manifold edges detected.")
        return None, None

    # Collect cells connected to the non-manifold edges
    collected_cell_ids = set()

    # For each edge (which is a line), get its points
    for edgeid in range(non_manifold_edges.n_cells):
        print("Edge : ", edgeid)
        # Extract the edge as a mesh
        edge_line = non_manifold_edges.extract_cells(edgeid)

        # Get point indices of this edge
        point_ids = map_edge_points_to_original_mesh(edge_line, mesh)
        print(f"Edge {edgeid} mapped original point IDs: {point_ids}")

        # For each point, find connected cells in the original mesh
        for cell_id in range(mesh.n_cells):
            pids = mesh.faces[4*cell_id+1:4*cell_id+4]  # Extract point IDs for the cell
            for pid in pids:
                if pid in point_ids:
                    collected_cell_ids.add(cell_id)

    collected_cell_ids = list(collected_cell_ids)

    print(f"Found {len(collected_cell_ids)} cells connected to non-manifold edges.")

    # Extract these cells from the original mesh
    print(np.array(collected_cell_ids).flatten())
    problematic_faces = mesh.extract_cells(np.array(collected_cell_ids).flatten())

    return problematic_faces, collected_cell_ids

def save_pyvista_as_legacy_vtk(vtk_mesh, mesh, output_filepath):
    # Extract points and faces from PyVista mesh
    points = mesh.points  # Nx3 numpy array
    faces = mesh.faces    # flat array: [n0, v0, v1, v2, n1, v0, v1, v2, ...] 
                        # where nX = number of points in face X (usually 3 or 4)

    # Create vtkPoints and set points
    vtk_points = vtk.vtkPoints()
    for p in points:
        vtk_points.InsertNextPoint(p)

    # Create vtkCellArray for faces
    vtk_faces = vtk.vtkCellArray()

    # PyVista's faces array encoding: first int = number of points in polygon, followed by the point indices
    i = 0
    while i < len(faces):
        n = faces[i]
        i += 1
        face_pts = faces[i:i+n]
        i += n
        vtk_faces.InsertNextCell(n)
        for pt_id in face_pts:
            vtk_faces.InsertCellPoint(pt_id)

    # Create vtkPolyData and set points and polys
    vtk_polydata = vtk.vtkPolyData()
    vtk_polydata.SetPoints(vtk_points)
    vtk_polydata.SetPolys(vtk_faces)
    transfer_all_scalar_arrays(vtk_mesh, vtk_polydata)
    # Now you have a proper vtkPolyData object!

    # Example: save with vtkPolyDataWriter
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_filepath)
    writer.SetInputData(vtk_polydata)
    writer.Write()

#--------------------------- Remove cells and save the mesh -----------------

def remove_problematic_cells_and_save(input_filepath, problematic_cell_ids, output_filepath, fill_holes=False, hole_size=5.0):
    """
    Remove problematic cells from the mesh and save the cleaned mesh.

    Parameters: 
    - input_filepath: path to the input mesh file (.vtk or .ply)
    - problematic_cell_ids: list of cell IDs to remove
    - output_filepath: path to save the cleaned mesh (.vtk or .ply)
    - fill_holes: boolean indicating whether to fill holes in the mesh (default: False)
    - hole_size: size of the holes to fill (default: 5.0 | empirical value, should not be higher unless your mesh is completely broken)
    """

    # Step 1: Open the mesh from the input file
    mesh = pv.read(input_filepath)

    #reading mesh in vtk format to transfer cell arrays 
    vtk_mesh = readvtk(input_filepath)

    if problematic_cell_ids is not None: 

        # Step 2: Create a mask for cells to keep
        n_cells = mesh.n_cells
        cell_mask = np.ones(n_cells, dtype=bool)
        cell_mask[problematic_cell_ids] = False

        # Step 3: Extract the clean mesh
        cleaned_mesh = mesh.extract_cells(cell_mask)

        # Step 4: If unstructured grid, convert to PolyData
        if isinstance(cleaned_mesh, pv.UnstructuredGrid):
            print("ℹ Converting UnstructuredGrid to PolyData before hole filling.")
            cleaned_mesh = cleaned_mesh.extract_surface()
        # Step 5: Clean degenerate elements
        cleaned_mesh = cleaned_mesh.clean()
        # Step 6: (Optional) Fill holes
        if fill_holes:
            cleaned_mesh = cleaned_mesh.fill_holes(hole_size=hole_size)

        # Step 7: Save the mesh
        save_pyvista_as_legacy_vtk(vtk_mesh, cleaned_mesh, output_filepath) #using vtk legacy format
        # cleaned_mesh.save(output_filepath)
        print(f" Mesh cleaned and saved to: {output_filepath}")

        return cleaned_mesh
    
    else :
        writevtk(vtk_mesh, output_filepath) #using vtk legacy format
        print(f"No problematic cells were removed, saving mesh to : {output_filepath}")
        return vtk_mesh

#main function 
def main(input_filepath, fill_holes=False, hole_size=5.0):
    """
    Main function to remove non-manifold edges from a mesh and save the cleaned mesh.

    Parameters:
    - input_filepath: path to the input mesh file (.vtk or .ply)    
    - fill_holes: boolean indicating whether to fill holes in the mesh (default: False)
    - hole_size: size of the holes to fill (default: 5.0 | empirical value, should not be higher unless your mesh is completely broken)
    """

    folder = os.path.dirname(input_filepath)
    filename = os.path.basename(input_filepath)
    name, extension = os.path.splitext(filename)

    output_filename = f"manifold_{name}{extension}"
    output_filepath = os.path.join(folder, output_filename)

    # Load the mesh 
    mesh = pv.read(input_filepath)  


    # Detect non-manifold edges and extract cells
    problematic_faces, collected_cell_ids = extract_cells_from_non_manifold_edges(input_filepath)

    if problematic_faces is not None:
        problematic_faces.plot()

    # Remove problematic cells and save the cleaned mesh
    remove_problematic_cells_and_save(input_filepath, collected_cell_ids, output_filepath, fill_holes=fill_holes, hole_size=hole_size)

    n_new_cells = detect_non_manifold_edges(output_filepath)[0]
    if n_new_cells == 0:
        print("No non-manifold edges detected in the cleaned mesh.")
    else:
        print(f"Warning: {n_new_cells} non-manifold edges detected in the cleaned mesh. Check if location is not on PVs")
    
    print(f"Cleaned mesh saved to: {output_filepath}")
    return 1



####### Launch in terminal (bash) ###########


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove non manifold edges of a mesh') 
    parser.add_argument('--input_filepath', type=str, help='Path to the input mesh file (.vtk or .ply)')
    parser.add_argument('--fill_holes', type=bool, default=True, help='Fill holes in the mesh (default: True) | Change this if you just want to delete non manifold edges') 
    parser.add_argument('--hole_size', type=float, default=5.0, help='Size of the holes to fill (default: 5.0 | empirical value, should not be higher unless your mesh is completely broken)')
    args = parser.parse_args()

    if os.path.isfile(args.input_filepath):
        print("Loading input mesh...")
        surface = pv.read(args.input_filepath)
        print(f"Mesh loaded: {surface.n_points} points, {surface.n_cells} cells")

    else:
        sys.exit('ERROR: input file does not exist. Please, specify a valid path.')

    main(args.input_filepath, args.fill_holes, args.hole_size)
