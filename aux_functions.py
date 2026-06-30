import vtk
import math
import numpy as np
from visualization_functions import *
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from scipy.spatial.distance import cdist
import scipy.sparse.linalg as linalg_sp
from scipy.sparse import hstack, coo_matrix, dia_matrix, vstack,  csc_matrix
import collections


###     Input/Output    ###
def readvtk(filename):
    """Read VTK file"""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def readvtp(filename):
    """Read VTP file"""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def writevtk(surface, filename, type='ascii'):
    """Write binary or ascii VTK file"""
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(surface)
    writer.SetFileName(filename)
    if type == 'ascii':
        writer.SetFileTypeToASCII()
    elif type == 'binary':
        writer.SetFileTypeToBinary()
    writer.Write()

def writevtp(surface, filename):
    """Write VTP file"""
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(surface)
    writer.SetFileName(filename)
    writer.Write()

###     Math    ###
def euclideandistance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def normvector(v):
    return math.sqrt(dot(v, v))

def angle(v1, v2):
    return math.acos(dot(v1, v2) / (normvector(v1) * normvector(v2)))

def acumvectors(point1, point2):
    return [point1[0] + point2[0], point1[1] + point2[1], point1[2] + point2[2]]

def subtractvectors(point1, point2):
    return [point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]]

def dividevector(point, n):
    nr = float(n)
    return [point[0]/nr, point[1]/nr, point[2]/nr]

def multiplyvector(point, n):
    nr = float(n)
    return [nr*point[0], nr*point[1], nr*point[2]]

def sumvectors(vect1, scalar, vect2):
    return [vect1[0] + scalar*vect2[0], vect1[1] + scalar*vect2[1], vect1[2] + scalar*vect2[2]]

def cross(v1, v2):
    return [v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]]

def dot(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def normalizevector(v):
    norm = normvector(v)
    return [v[0] / norm, v[1] / norm, v[2] / norm]

###     Mesh processing     ###

def cleanpolydata(polydata):
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.Update()
    return cleaner.GetOutput()

def fillholes(polydata, size):
    """Fill mesh holes smaller than 'size' """
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(polydata)
    filler.SetHoleSize(size)
    filler.Update()
    return filler.GetOutput()

def pointthreshold(polydata, arrayname, start=0, end=1, alloff=0):
    """ Clip polydata according to given thresholds in scalar array"""
    threshold = vtk.vtkThreshold()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 9):
        threshold.SetLowerThreshold(start)
        threshold.SetUpperThreshold(end)
    else:
        threshold.ThresholdBetween(start, end)

    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, arrayname)
    if (alloff):
        threshold.AllScalarsOff()
    threshold.Update()
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(threshold.GetOutput())
    surfer.Update()
    return surfer.GetOutput()

def cellthreshold(polydata, arrayname, start=0, end=1):
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, arrayname)
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 9):
        threshold.SetLowerThreshold(start)
        threshold.SetUpperThreshold(end)
    else:
        threshold.ThresholdBetween(start, end)
    threshold.Update()

    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputConnection(threshold.GetOutputPort())
    surfer.Update()
    return surfer.GetOutput()

def roundpointarray(polydata, name):
    """Round values in point array"""
    # get original array
    array = polydata.GetPointData().GetArray(name)
    # round labels
    for i in range(polydata.GetNumberOfPoints()):
        value = array.GetValue(i)
        array.SetValue(i, round(value))
    return polydata

def planeclip(surface, point, normal, insideout=1):
    """Clip surface using plane given by point and normal"""
    clipplane = vtk.vtkPlane()
    clipplane.SetOrigin(point)
    clipplane.SetNormal(normal)
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(surface)
    clipper.SetClipFunction(clipplane)

    if insideout == 1:
        clipper.InsideOutOn()
    else:
        clipper.InsideOutOff()
    clipper.Update()
    return clipper.GetOutput()

def sphereclip(dataset, center, radius,
               shrink_factor=1.1,
               max_iterations=10):
    """
    Clip the dataset with a sphere of given center and radius.
    Keeps only the part inside the sphere.
    """
    sphere = vtk.vtkSphere()
    sphere.SetCenter(center)
    sphere.SetRadius(radius)

    clipper = vtk.vtkClipPolyData()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        clipper.SetInputData(dataset)
    else:
        clipper.SetInput(dataset)
    clipper.SetClipFunction(sphere)
    clipper.SetInsideOut(False)  # Keep outside the sphere
    clipper.Update()

    final_surface = extractlargestregion(clipper.GetOutput())
    edges = extractboundaryedge(final_surface)
    conn = get_connected_edges(edges)
    mesh_holes_after = conn.GetNumberOfExtractedRegions()
    edges = extractboundaryedge(dataset)
    conn = get_connected_edges(edges)
    mesh_holes_before = conn.GetNumberOfExtractedRegions()
    i=0
    while (mesh_holes_after - mesh_holes_before > 1) and i<5:
        print('Multiple contours detected, increasing radius')
        sphere = vtk.vtkSphere()
        sphere.SetCenter(center)
        sphere.SetRadius(radius*shrink_factor)

        clipper = vtk.vtkClipPolyData()
        if vtk.vtkVersion.GetVTKMajorVersion() > 5:
            clipper.SetInputData(dataset)
        else:
            clipper.SetInput(dataset)
        clipper.SetClipFunction(sphere)
        clipper.SetInsideOut(False)  # Keep outside the sphere
        clipper.Update()

        final_surface = extractlargestregion(clipper.GetOutput())
        edges = extractboundaryedge(final_surface)
        conn = get_connected_edges(edges)
        mesh_holes_after = conn.GetNumberOfExtractedRegions()
        edges = extractboundaryedge(dataset)
        conn = get_connected_edges(edges)
        mesh_holes_before = conn.GetNumberOfExtractedRegions()
        i+=1
    return final_surface

def cutdataset(dataset, point, normal):
    """Similar to planeclip but using vtkCutter instead of vtkClipPolyData"""
    cutplane = vtk.vtkPlane()
    cutplane.SetOrigin(point)
    cutplane.SetNormal(normal)
    cutter = vtk.vtkCutter()
    cutter.SetInputData(dataset)
    cutter.SetCutFunction(cutplane)
    cutter.Update()
    return cutter.GetOutput()

def pointset_centreofmass(polydata):
    centre = [0, 0, 0]
    for i in range(polydata.GetNumberOfPoints()):
        point = [polydata.GetPoints().GetPoint(i)[0],
          polydata.GetPoints().GetPoint(i)[1],
          polydata.GetPoints().GetPoint(i)[2]]
        centre = acumvectors(centre,point)
    return dividevector(centre, polydata.GetNumberOfPoints())

def seeds_to_csv(seedsfile, arrayname, labels, outfile):
    """Read seeds from VTP file, write coordinates in csv"""
    f = open(outfile, 'w')
    allseeds = readvtp(seedsfile)
    for l in labels:
        currentseeds = pointthreshold(allseeds, arrayname, l, l, 0)
        currentpoint = pointset_centreofmass(currentseeds)

        line = str(currentpoint[0]) + ',' + str(currentpoint[1]) + ',' + str(currentpoint[2]) + '\n'
        f.write(line)
    f.close()

def generateglyph(polyIn, scalefactor=2):
    vertexGlyphFilter = vtk.vtkGlyph3D()
    sphereSource = vtk.vtkSphereSource()
    vertexGlyphFilter.SetSourceData(sphereSource.GetOutput())
    vertexGlyphFilter.SetInputData(polyIn)
    vertexGlyphFilter.SetColorModeToColorByScalar()
    vertexGlyphFilter.SetSourceConnection(sphereSource.GetOutputPort())
    vertexGlyphFilter.ScalingOn()
    vertexGlyphFilter.SetScaleFactor(scalefactor)
    vertexGlyphFilter.Update()
    return vertexGlyphFilter.GetOutput()

def extractboundaryedge(polydata):
    edge = vtk.vtkFeatureEdges()
    edge.SetInputData(polydata)
    edge.BoundaryEdgesOn() 
    edge.FeatureEdgesOff()
    edge.NonManifoldEdgesOff()
    edge.Update()
    return edge.GetOutput()

def extractlargestregion(polydata):
    """Keep only biggest region"""
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surfer.GetOutput())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputData(cleaner.GetOutput())
    connect.SetExtractionModeToLargestRegion()
    connect.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(connect.GetOutput())
    cleaner.Update()
    return cleaner.GetOutput()

def extract_all_regions(polydata):
    """
    Returns a list of vtkPolyData objects, each corresponding
    to one connected component of the input polydata.
    """
    # Clean and get surface
    surf = vtk.vtkDataSetSurfaceFilter()
    surf.SetInputData(polydata)
    surf.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(surf.GetOutput())
    clean.Update()

    # Connectivity filter
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(clean.GetOutput())
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()  # assigns "RegionId"
    conn.Update()

    output = conn.GetOutput()
    region_ids = output.GetCellData().GetArray("RegionId")
    n_regions = conn.GetNumberOfExtractedRegions()

    regions = []
    max_n_cells = 0
    for rid in range(n_regions):
        # Extract one region at a time
        conn2 = vtk.vtkPolyDataConnectivityFilter()
        conn2.SetInputData(output)
        conn2.SetExtractionModeToSpecifiedRegions()
        conn2.InitializeSpecifiedRegionList()
        conn2.AddSpecifiedRegion(rid)
        conn2.Update()

        clean2 = vtk.vtkCleanPolyData()
        clean2.SetInputData(conn2.GetOutput())
        clean2.Update()
        reg = clean2.GetOutput()
        if reg.GetNumberOfCells()>max_n_cells:
            max_n_cells = reg.GetNumberOfCells()
        regions.append(reg)

    return regions, max_n_cells

def countregions(polydata):
    """Count number of connected components/regions"""
    # preventive measures: clean before connectivity filter to avoid artificial regionIds
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surfer.GetOutput())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputData(cleaner.GetOutput())
    connect.Update()
    return connect.GetNumberOfExtractedRegions()

def extractclosestpointregion(polydata, point=[0, 0, 0]):
    # preventive measures: clean before connectivity filter to avoid artificial regionIds. It slices the surface down the middle
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surfer.GetOutput())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()

    connect.SetInputData(cleaner.GetOutput())
    connect.SetExtractionModeToClosestPointRegion()
    connect.SetClosestPoint(point)
    connect.Update()
    return connect.GetOutput()

def extractconnectedregion(polydata, regionid):
    """Extract connected region with label = regionid """
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surfer.GetOutput())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputData(cleaner.GetOutput())

    connect.SetExtractionModeToAllRegions()
    connect.ColorRegionsOn()
    connect.Update()
    surface = pointthreshold(connect.GetOutput(), 'RegionId', float(regionid), float(regionid))
    return surface

def get_connected_edges(polydata):
    """Extract all connected regions"""
    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputData(polydata)
    connect.SetExtractionModeToAllRegions()
    connect.ColorRegionsOn()
    connect.Update()
    return connect

def find_create_path(mesh, p1, p2):
    """Get shortest path (using Dijkstra algorithm) between p1 and p2 on the mesh. Returns a polydata"""
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    # (VTK 9.1 and later...) The Dijkistra interpolator will not accept cells that aren't triangles
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 9):
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(mesh)
        triangleFilter.Update()
        pd = triangleFilter.GetOutput()
        dijkstra.SetInputData(pd)
    else:
        dijkstra.SetInputData(mesh)

    dijkstra.SetStartVertex(p1)
    dijkstra.SetEndVertex(p2)
    dijkstra.Update()
    return dijkstra.GetOutput()

def min_euclidean_distance_fast(points1, points2):
    dists = cdist(points1, points2)
    i, j = np.unravel_index(np.argmin(dists), dists.shape)
    return dists[i, j], i, j

def find_create_path_contours(mesh, c1, c2, c3=[]):
    if len(c3)>0:
        dist, best_id1a, best_id2 = min_euclidean_distance_fast(vtk_to_numpy(mesh.GetPoints().GetData())[c1, :], vtk_to_numpy(mesh.GetPoints().GetData())[c2, :])
        dist, best_id1b, _ = min_euclidean_distance_fast(vtk_to_numpy(mesh.GetPoints().GetData())[c1, :], vtk_to_numpy(mesh.GetPoints().GetData())[c3, :])

        if abs(best_id1a - best_id1b)<10:
            best_id1 = int((best_id1a  + best_id1b)/2)
        else:
            best_id1 = best_id1a 
    else:
        dist, best_id1, best_id2 = min_euclidean_distance_fast(vtk_to_numpy(mesh.GetPoints().GetData())[c1, :], vtk_to_numpy(mesh.GetPoints().GetData())[c2, :])
    best_path = find_create_path(mesh, c1[best_id1], c2[best_id2])

    return best_path

def update_same_endpoint(pathA, pathB, pathA_ids, pathB_ids, common_cont_ids, index_A, index_A_cont, index_B, index_B_cont):
    new_id = int(np.where(common_cont_ids == pathA_ids[index_A_cont])[0])+1
    pathA_temp = find_create_path(m_open, common_cont_ids[new_id], pathA_ids[index_A])
    pathA_ids_temp = get_ids(pathA_temp, locator_open).astype(int)
    if paths_intersect(pathA_ids_temp, pathB_ids):
        new_id = int(np.where(common_cont_ids == pathB_ids[index_B_cont])[0])+1
        pathB = find_create_path(m_open, common_cont_ids[new_id], pathB_ids[index_B])
    else:
        pathA = pathA_temp
    return pathA, pathB
        
def adjust_path(pathA, pathB, pathA_ids, pathB_ids, common_cont_ids):
    if pathA_ids[0] in common_cont_ids:
        if pathB_ids[0] in common_cont_ids:
            if pathA_ids[0] == pathB_ids[0]:
                pathA, pathB = update_same_endpoint(pathA, pathB, pathA_ids, pathB_ids, common_cont_ids, -1, 0, -1, 0)
            else:
                pathA = find_create_path(m_open, pathA_ids[-1], pathB_ids[0])
                pathB = find_create_path(m_open, pathB_ids[-1], pathA_ids[0])
        elif pathB_ids[-1] in common_cont_ids:
            if pathA_ids[0] == pathB_ids[-1]:
                pathA, pathB = update_same_endpoint(pathA, pathB, pathA_ids, pathB_ids, common_cont_ids, -1, 0, -0, -1)
            else:
                pathA = find_create_path(m_open, pathA_ids[-1], pathB_ids[-1])
                pathB = find_create_path(m_open, pathA_ids[0], pathB_ids[0])
    elif pathA_ids[-1] in common_cont_ids:
        if pathB_ids[0] in common_cont_ids:
            if pathA_ids[-1] == pathB_ids[0]:
                pathA, pathB = update_same_endpoint(pathA, pathB, pathA_ids, pathB_ids, common_cont_ids, 0, -1, -1, 0)
            else:
                pathA = find_create_path(m_open, pathB_ids[0], pathA_ids[0])
                pathB = find_create_path(m_open, pathB_ids[-1], pathA_ids[-1])
        elif pathB_ids[-1] in common_cont_ids:
            if pathA_ids[-1] == pathB_ids[-1]:
                pathA, pathB = update_same_endpoint(pathA, pathB, pathA_ids, pathB_ids, common_cont_ids, 0, -1, 0, -1)
            else:
                pathA = find_create_path(m_open, pathB_ids[-1], pathA_ids[0])
                pathB = find_create_path(m_open, pathA_ids[-1], pathB_ids[0])
    return pathA, pathB

def check_and_adjust_paths(locator_open, path1, path2, path3, path4, path5, path6, path7, path8a, path8b, path8c,
                            ripv_cont_ids, rspv_cont_ids, lipv_cont_ids, lspv_cont_ids, laa_cont_ids):
    path1_ids = get_ids(path1, locator_open).astype(int)
    path2_ids = get_ids(path2, locator_open).astype(int)
    path3_ids = get_ids(path3, locator_open).astype(int)
    path4_ids = get_ids(path4, locator_open).astype(int)
    path5_ids = get_ids(path5, locator_open).astype(int)
    path6_ids = get_ids(path6, locator_open).astype(int)
    path7_ids = get_ids(path7, locator_open).astype(int)   
    path8a_ids = get_ids(path8a, locator_open).astype(int)  
    path8b_ids = get_ids(path8b, locator_open).astype(int)                                                                                                                                                                                                                                                                                                                                        
    path8c_ids = get_ids(path8c, locator_open).astype(int)                                                                                                                                                                                                                                                                                                                                          

    if paths_intersect(path1_ids, path2_ids):
        print("Overlap detected between paths 1 and 2, which have the RIPV as common contour.")
        path1, path2 = adjust_path(path1, path2, path1_ids, path2_ids, ripv_cont_ids)
    if paths_intersect(path1_ids, path4_ids):
        print("Overlap detected between paths 1 and 4, which have the RSPV as common contour.")
        path1, path4 = adjust_path(path1, path4, path1_ids, path4_ids, rspv_cont_ids)
    if paths_intersect(path1_ids, path5_ids):
        print("Overlap detected between paths 1 and 5, which have the RSPV as common contour.")
        path1, path5 = adjust_path(path1, path5, path1_ids, path5_ids, rspv_cont_ids)
    if paths_intersect(path1_ids, path6_ids):
        print("Overlap detected between paths 1 and 6, which have the RIPV as common contour.")
        path1, path6 = adjust_path(path1, path6, path1_ids, path6_ids, ripv_cont_ids)
    if paths_intersect(path2_ids, path3_ids):
        print("Overlap detected between paths 2 and 3, which have the LIPV as common contour.")
        path2, path3 = adjust_path(path2, path3, path2_ids, path3_ids, lipv_cont_ids)
    if paths_intersect(path2_ids, path6_ids):
        print("Overlap detected between paths 2 and 6, which have the RIPV as common contour.")
        path2, path6 = adjust_path(path2, path6, path2_ids, path6_ids, ripv_cont_ids)
    if paths_intersect(path2_ids, path7_ids):
        print("Overlap detected between paths 2 and 7, which have the LIPV as common contour.")
        path2, path7 = adjust_path(path2, path7, path2_ids, path7_ids, lipv_cont_ids)
    if paths_intersect(path3_ids, path4_ids):
        print("Overlap detected between paths 3 and 4, which have the LSPV as common contour.")
        path3, path4 = adjust_path(path3, path4, path3_ids, path4_ids, lspv_cont_ids)
    if paths_intersect(path3_ids, path7_ids):
        print("Overlap detected between paths 3 and 7, which have the LIPV as common contour.")
        path3, path7 = adjust_path(path3, path7, path3_ids, path7_ids, lipv_cont_ids)
    if paths_intersect(path3_ids, path8a_ids):
        print("Overlap detected between paths 3 and 8a, which have the LSPV as common contour.")
        path3, path8a = adjust_path(path3, path8a, path3_ids, path8a_ids, lspv_cont_ids)
    if paths_intersect(path4_ids, path5_ids):
        print("Overlap detected between paths 4 and 5, which have the RSPV as common contour.")
        path4, path5 = adjust_path(path4, path5, path4_ids, path5_ids, rspv_cont_ids)
    if paths_intersect(path4_ids, path8a_ids):
        print("Overlap detected between paths 4 and 8a, which have the LSPV as common contour.")
        path4, path8a = adjust_path(path4, path8a, path4_ids, path8a_ids, lspv_cont_ids)
    if paths_intersect(path4_ids, path8c_ids):
        print("Overlap detected between paths 4 and 8c, which have the RSPV as common contour.")
        path4, path8c = adjust_path(path4, path8c, path4_ids, path8c_ids, rspv_cont_ids)
    if paths_intersect(path5_ids, path8c_ids):
        print("Overlap detected between paths 5 and 8c, which have the RSPV as common contour.")
        path5, path8c = adjust_path(path5, path8c, path5_ids, path8c_ids, rspv_cont_ids)
    if paths_intersect(path8a_ids, path8b_ids):
        print("Overlap detected between paths 8a and 8b, which have the LAA as common contour.")
        path8a, path8b = adjust_path(path8a, path8b, path8a_ids, path8b_ids, laa_cont_ids)
    if paths_intersect(path8a_ids, path8c_ids):
        print("Overlap detected between paths 8a and 8c, which have the LAA as common contour.")
        path8a, path8c = adjust_path(path8a, path8c, path8a_ids, path8c_ids, laa_cont_ids)
    if paths_intersect(path8b_ids, path8c_ids):
        print("Overlap detected between paths 8b and 8c, which have the LAA as common contour.")
        path8b, path8c = adjust_path(path8b, path8c, path8b_ids, path8c_ids, laa_cont_ids)
    
    return path1, path2, path3, path4, path5, path6, path7, path8a, path8b, path8c

def find_trimmed_path_between_contours(mesh, p1, p2, contourA_ids, contourB_ids):
    """
    Compute shortest path on mesh from p1 to p2,
    then trim it so it starts at first intersection with contour A
    and ends at first intersection with contour B.

    Returns: vtkPolyData (polyline)
    """
    # --- Run Dijkstra ---
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()

    if vtk.vtkVersion.GetVTKMajorVersion() >= 9:
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(mesh)
        triangleFilter.Update()
        pd = triangleFilter.GetOutput()
        dijkstra.SetInputData(pd)
    else:
        pd = mesh
        dijkstra.SetInputData(mesh)

    dijkstra.SetStartVertex(p2)
    dijkstra.SetEndVertex(p1)
    dijkstra.Update()

    path_pd = dijkstra.GetOutput()

    # --- Extract original mesh point IDs from path ---
    path_points = path_pd.GetPoints()
    path_ids = []

    # For each point in the path, find its ID in the original mesh
    mesh_points = mesh.GetPoints()
    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(mesh)
    point_locator.BuildLocator()

    for i in range(path_points.GetNumberOfPoints()):
        pt = path_points.GetPoint(i)
        original_id = point_locator.FindClosestPoint(pt)
        path_ids.append(original_id)
    contourA = set(contourA_ids)
    contourB = set(contourB_ids)

    # --- Find first hit on contour A ---
    start_idx = None
    for i, pid in enumerate(path_ids):
        if pid in contourA:
            start_idx = i
    # --- Find first hit on contour B after A ---
    end_idx = None
    if start_idx is not None:
        for i in range(start_idx, len(path_ids)):
            if path_ids[i] in contourB:
                end_idx = i
                break
    # --- Fallbacks ---
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(path_ids) - 1

    if start_idx != 0 or end_idx != len(path_ids) - 1 :
        dijkstra.SetStartVertex(path_ids[end_idx])
        dijkstra.SetEndVertex(path_ids[start_idx])
        dijkstra.Update()

        path_pd = dijkstra.GetOutput()
        
    return path_pd, path_ids[start_idx], path_ids[end_idx]

def compute_geodesic_distance(mesh, id_p1, id_p2):
    """Compute geodesic distance from point id_p1 to id_p2 on surface 'mesh'
    It first computes the path across the edges and then the corresponding distance adding up point to point distances)"""
    path = find_create_path(mesh, id_p1, id_p2)
    total_dist = 0
    n = path.GetNumberOfPoints()
    for i in range(n-1):   # Ids are ordered in the new polydata, from 0 to npoints_in_path
        p0 = path.GetPoint(i)
        p1 = path.GetPoint(i+1)
        dist = math.sqrt(math.pow(p0[0]-p1[0], 2) + math.pow(p0[1]-p1[1], 2) + math.pow(p0[2]-p1[2], 2) )
        total_dist = total_dist + dist
    return total_dist, path

def transfer_array(ref, target, arrayname, targetarrayname):
    """Transfer scalar array using closest point approximation"""
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(ref)
    locator.BuildLocator()

    refarray = ref.GetPointData().GetArray(arrayname)  # get array from reference

    numberofpoints = target.GetNumberOfPoints()
    newarray = vtk.vtkDoubleArray()
    newarray.SetName(targetarrayname)
    newarray.SetNumberOfTuples(numberofpoints)
    target.GetPointData().AddArray(newarray)

    # go through each point of target surface, determine closest point on surface, copy value
    for i in range(target.GetNumberOfPoints()):
        point = target.GetPoint(i)
        closestpoint_id = locator.FindClosestPoint(point)
        value = refarray.GetValue(closestpoint_id)
        newarray.SetValue(i, value)
    return target

def transfer_all_scalar_arrays(m1, m2):
    """ Transfer all scalar arrays from m1 to m2"""
    for i in range(m1.GetPointData().GetNumberOfArrays()):
        print('Transferring scalar array: {}'.format(m1.GetPointData().GetArray(i).GetName()))
        transfer_array(m1, m2, m1.GetPointData().GetArray(i).GetName(), m1.GetPointData().GetArray(i).GetName())

def transfer_all_scalar_arrays_by_point_id(m1, m2):
    """ Transfer all scalar arrays from m1 to m2 by point id"""
    for i in range(m1.GetPointData().GetNumberOfArrays()):
        print('Transferring scalar array: {}'.format(m1.GetPointData().GetArray(i).GetName()))
        m2.GetPointData().AddArray(m1.GetPointData().GetArray(i))

def get_ordered_cont_ids_based_on_distance(mesh):
    """ Given a contour, get the ordered list of Ids (not ordered by default).
    Open the mesh duplicating the point with id = 0. Compute distance transform of point 0
    and get a ordered list of points (starting in 0) """
    m = vtk.vtkMath()
    m.RandomSeed(0)
    # copy the original mesh point by point
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    cover = vtk.vtkPolyData()
    nver = mesh.GetNumberOfPoints()
    points.SetNumberOfPoints(nver+1)

    new_pid = nver  # id of the duplicated point
    added = False

    for j in range(mesh.GetNumberOfCells()):
        # get the 2 point ids
        ptids = mesh.GetCell(j).GetPointIds()
        cell = mesh.GetCell(j)
        if (ptids.GetNumberOfIds() != 2):
            print("Non contour mesh (lines)")
            break

        # read the 2 involved points
        pid0 = ptids.GetId(0)
        pid1 = ptids.GetId(1)
        p0 = mesh.GetPoint(ptids.GetId(0))   # returns coordinates
        p1 = mesh.GetPoint(ptids.GetId(1))
        if pid0 == 0:
            if added == False:
                # Duplicate point 0. Add gaussian noise to the original point
                new_p = [p0[0] + m.Gaussian(0.0, 0.0005), p0[1] + m.Gaussian(0.0, 0.0005), p0[2] + m.Gaussian(0.0, 0.0005)]
                points.SetPoint(new_pid, new_p)
                points.SetPoint(pid1, p1)
                polys.InsertNextCell(2)
                polys.InsertCellPoint(pid1)
                polys.InsertCellPoint(new_pid)
                added = True
            else:  # act normal
                points.SetPoint(ptids.GetId(0), p0)
                points.SetPoint(ptids.GetId(1), p1)
                polys.InsertNextCell(2)
                polys.InsertCellPoint(cell.GetPointId(0))
                polys.InsertCellPoint(cell.GetPointId(1))
        elif pid1 == 0:
            if added == False:
                new_p = [p1[0] + m.Gaussian(0.0, 0.0005), p1[1] + m.Gaussian(0.0, 0.0005), p1[2] + m.Gaussian(0.0, 0.0005)]
                points.SetPoint(new_pid, new_p)
                points.SetPoint(pid0, p0)
                polys.InsertNextCell(2)
                polys.InsertCellPoint(pid0)
                polys.InsertCellPoint(new_pid)
                added = True
            else:  # act normal
                points.SetPoint(ptids.GetId(0), p0)
                points.SetPoint(ptids.GetId(1), p1)
                polys.InsertNextCell(2)
                polys.InsertCellPoint(cell.GetPointId(0))
                polys.InsertCellPoint(cell.GetPointId(1))

        else:
            points.SetPoint(ptids.GetId(0), p0)
            points.SetPoint(ptids.GetId(1), p1)
            polys.InsertNextCell(2)
            polys.InsertCellPoint(cell.GetPointId(0))
            polys.InsertCellPoint(cell.GetPointId(1))

    if added == False:
        print('Warning: I have not added any point, list of indexes may not be correct.')
    cover.SetPoints(points)

    if (vtk.vtkVersion.GetVTKMajorVersion() >= 9):
        cover.SetLines(polys)
    else:
        cover.SetPolys(polys)
    # compute distance from point with id 0 to all the rest
    npoints = cover.GetNumberOfPoints()
    dists = np.zeros(npoints)
    for i in range(1,npoints):
        [dists[i], poly] = compute_geodesic_distance(cover, int(0), i)
    list_ = np.argsort(dists).astype(int)
    return list_[0:len(list_)-1]    # skip last one, duplicated

def define_pv_segments_proportions(t_v5_1, t_v5_2, t_v6, t_v7, alpha):
    """define number of points of each pv hole segment to ensure appropriate distribution"""
    props = np.zeros([5, 4])
    props[0, 0] = np.divide(1.0, 4.0)  
    props[0, 1] = t_v5_2 * np.divide(1.0, 2.0*np.pi)
    props[0, 2] = np.divide(1.0, 4.0) + t_v5_1 * np.divide(1.0, 2.0*np.pi)
    props[0, 3] = 1.0 - props[0, 0] - props[0, 1] - props[0, 3]

    props[1, 0] = np.divide(t_v6, 2.0*np.pi) - np.divide(1.0, 2.0)
    props[1, 2] = np.divide(1.0, 4.0)  # s3
    props[1, 1] = 1.0 - props[1, 0] - props[1, 2]   # s2

    props[2, 2] = np.divide(1.0, 4.0)
    props[2, 0] = np.divide(t_v7, 2.0*np.pi) - np.divide(1.0, 4.0)
    props[2, 1] = 1.0 - props[2, 0] - props[2, 2]

    props[3, 0] = np.divide(1.0, 4.0)   
    props[3, 1] = np.divide(1.0, 2.0)  
    props[3, 2] = np.divide(1.0, 4.0)

    props[4, 0] = np.divide(3.0, 8.0)
    props[4, 1] = np.divide(1.0, 2.0)
    props[4, 2] = np.divide(1.0, 8.0)
    return props

def define_mv_segments_proportions():
    """define number of points of each mv hole segment to ensure appropriate distribution"""
    props = np.zeros(4)
    props[0] = 0.19                              # Septal wall
    props[1] = 0.3                              # inferior wall
    props[2] = 0.19                             # lateral wall
    props[3] = 1.0 - np.sum(props[0:3])         # anterior wall
    return props

def define_disk_template(rdisk, rhole_rspv, rhole_ripv, rhole_lipv, rhole_lspv, rhole_laa, xhole_center, yhole_center,
                         laa_hole_center_x, laa_hole_center_y, t_v5_1, t_v5_2, t_v6, t_v7, t_v8):
    """Define target positions in the disk template, return coordinates (x,y) corresponding to:
    v1r, v1d, v1l, v2u, v2r, v2l, v3u, v3r, v3l, v4r, v4u, v4d, vlaad, vlaau, p5, p6, p7, p8 """
    coordinates = np.zeros([2, 20])
    complete_circumf_t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    rspv_hole_x = np.cos(complete_circumf_t) * rhole_rspv + xhole_center[0]
    rspv_hole_y = np.sin(complete_circumf_t) * rhole_rspv + yhole_center[0]
    ripv_hole_x = np.cos(complete_circumf_t) * rhole_ripv + xhole_center[1]
    ripv_hole_y = np.sin(complete_circumf_t) * rhole_ripv + yhole_center[1]
    lipv_hole_x = np.cos(complete_circumf_t) * rhole_lipv + xhole_center[2]
    lipv_hole_y = np.sin(complete_circumf_t) * rhole_lipv + yhole_center[2]
    lspv_hole_x = np.cos(complete_circumf_t) * rhole_lspv + xhole_center[3]
    lspv_hole_y = np.sin(complete_circumf_t) * rhole_lspv + yhole_center[3]
    laa_hole_x = np.cos(complete_circumf_t) * rhole_laa + laa_hole_center_x
    laa_hole_y = np.sin(complete_circumf_t) * rhole_laa + laa_hole_center_y
    # define (x,y) positions where I put v5, v6, v7 and v8
    coordinates[0, 14] = np.cos(t_v5_1) * rdisk  # p5_x
    coordinates[1, 14] = np.sin(t_v5_1) * rdisk  # p5_y
    coordinates[0, 15] = np.cos(t_v6) * rdisk  # p6_x
    coordinates[1, 15] = np.sin(t_v6) * rdisk  # p6_y
    coordinates[0, 16] = np.cos(t_v7) * rdisk  # p7_x
    coordinates[1, 16] = np.sin(t_v7) * rdisk  # p7_y
    coordinates[0, 17] = np.cos(t_v8) * rdisk  # p8_x
    coordinates[1, 17] = np.sin(t_v8) * rdisk  # p8_y

    # define target points corresponding to the pv holes
    # RSPV (right (in the line connecting to MV), left (horizontal line), down (vertical line), up (connecting to LAA))
    coordinates[0, 0] = rspv_hole_x[np.abs(complete_circumf_t - t_v5_1).argmin()]   # v1r_x, x in rspv circumf where angle is pi/4
    coordinates[1, 0] = rspv_hole_y[np.abs(complete_circumf_t - t_v5_1).argmin()]   # v1r_y
    coordinates[0, 1] = rspv_hole_x[np.abs(complete_circumf_t - (3 * np.pi / 2)).argmin()]
    coordinates[1, 1] = rspv_hole_y[np.abs(complete_circumf_t - (3 * np.pi / 2)).argmin()]
    coordinates[0, 2] = rspv_hole_x[(np.abs(complete_circumf_t - np.pi)).argmin()]
    coordinates[1, 2] = rspv_hole_y[(np.abs(complete_circumf_t - np.pi)).argmin()]
    coordinates[0, 18] = rspv_hole_x[(np.abs(complete_circumf_t- (np.pi - t_v5_2))).argmin()]
    coordinates[1, 18] = rspv_hole_y[(np.abs(complete_circumf_t - (np.pi - t_v5_2))).argmin()]
    # RIPV
    coordinates[0, 3] = ripv_hole_x[np.abs(complete_circumf_t - (np.pi / 2)).argmin()]  # x in ripv circumf UP
    coordinates[1, 3] = ripv_hole_y[np.abs(complete_circumf_t - (np.pi / 2)).argmin()]
    coordinates[0, 4] = ripv_hole_x[np.abs(complete_circumf_t - t_v6).argmin()]
    coordinates[1, 4] = ripv_hole_y[np.abs(complete_circumf_t - t_v6).argmin()]
    coordinates[0, 5] = ripv_hole_x[np.abs(complete_circumf_t - (np.pi)).argmin()]
    coordinates[1, 5] = ripv_hole_y[np.abs(complete_circumf_t - (np.pi)).argmin()]
    # LIPV
    coordinates[0, 6] = lipv_hole_x[np.abs(complete_circumf_t - (np.pi / 2)).argmin()]
    coordinates[1, 6] = lipv_hole_y[np.abs(complete_circumf_t - (np.pi / 2)).argmin()]
    coordinates[0, 7] = lipv_hole_x[complete_circumf_t.argmin()]  # angle = 0
    coordinates[1, 7] = lipv_hole_y[complete_circumf_t.argmin()]
    coordinates[0, 8] = lipv_hole_x[np.abs(complete_circumf_t - t_v7).argmin()]
    coordinates[1, 8] = lipv_hole_y[np.abs(complete_circumf_t - t_v7).argmin()]
    # LSPV
    coordinates[0, 9] = lspv_hole_x[complete_circumf_t.argmin()]  # angle = 0
    coordinates[1, 9] = lspv_hole_y[complete_circumf_t.argmin()]
    coordinates[0, 10] = lspv_hole_x[np.abs(complete_circumf_t - (np.pi / 2)).argmin()]
    coordinates[1, 10] = lspv_hole_y[np.abs(complete_circumf_t - (np.pi / 2)).argmin()]
    coordinates[0, 11] = lspv_hole_x[np.abs(complete_circumf_t - (3 * np.pi / 2)).argmin()]
    coordinates[1, 11] = lspv_hole_y[np.abs(complete_circumf_t - (3 * np.pi / 2)).argmin()]
    # LAA
    coordinates[0, 12] = laa_hole_x[np.abs(complete_circumf_t - (3 * np.pi / 2)).argmin()]
    coordinates[1, 12] = laa_hole_y[np.abs(complete_circumf_t - (3 * np.pi / 2)).argmin()]
    coordinates[0, 13] = laa_hole_x[np.abs(complete_circumf_t - t_v8).argmin()]  # angle = pi/2 + pi/8
    coordinates[1, 13] = laa_hole_y[np.abs(complete_circumf_t - t_v8).argmin()]
    coordinates[0, 19] = laa_hole_x[np.abs(complete_circumf_t).argmin()]
    coordinates[1, 19] = laa_hole_y[np.abs(complete_circumf_t).argmin()]
    return coordinates

def get_coords(c):
    """Given all coordinates in a matrix, identify and return them separately"""
    return c[0,0], c[1,0], c[0,1], c[1,1], c[0,2], c[1,2], c[0,3], c[1,3], c[0,4], c[1,4], c[0,5], c[1,5], c[0,6], c[1,6], c[0,7], c[1,7], c[0,8], c[1,8], c[0,9], c[1,9], c[0,10], c[1,10], c[0,11], c[1,11], c[0,12], c[1,12], c[0,13], c[1,13], c[0,14], c[1,14], c[0,15], c[1,15], c[0,16], c[1,16], c[0,17], c[1,17], c[0,18], c[1,18], c[0,19], c[1,19]

def extract_LA_contours(m_open, filename, m_whole, save=False):
    """Given LA with clipped PVs, LAA and MV identify and classify all 5 contours using 'autolabels' array.
    Save contours if save=True"""
    edges = extractboundaryedge(m_open)
    conn = get_connected_edges(edges)
    poly_edges = conn.GetOutput()
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(m_whole)
    locator.BuildLocator()
    if save==True:
        writevtk(poly_edges, filename[0:len(filename) - 4] + '_detected_edges.vtk')

    print('Detected {} regions'.format(conn.GetNumberOfExtractedRegions()))
    if conn.GetNumberOfExtractedRegions() != 6:
        print(
            'WARNING: the number of contours detected is not the expected. The classification of contours may be wrong')
                        
    autolabels_full = vtk_to_numpy(m_whole.GetPointData().GetArray('autolabels'))
    for i in range(conn.GetNumberOfExtractedRegions()):
        print('Detecting region index: {}'.format(i))
        c = pointthreshold(poly_edges, 'RegionId', i, i)
        
        autolabels = vtk_to_numpy(c.GetPointData().GetArray('autolabels'))
        mostcommonlist = collections.Counter(autolabels.astype(int)).most_common()
        if len(mostcommonlist) != 1:
            if mostcommonlist[0][0] != 36:
                mostcommon = mostcommonlist[0][0]
                for id in get_ids(c, locator).astype(int):
                    autolabels_full[id] = mostcommon 
            else:
                mostcommon = mostcommonlist[1][0]
                for id in get_ids(c, locator).astype(int):
                    autolabels_full[id] = mostcommon 
            autolabels_full_array = numpy_to_vtk(autolabels_full)
            autolabels_full_array.SetName('autolabels')
            m_whole.GetPointData().AddArray(autolabels_full_array)
            writevtp(m_whole, filename[0:len(filename) - 14] + '_autolabels.vtp')
        else:
            mostcommon = mostcommonlist[0][0]
        if mostcommon == 36:  # use the most repeated label since some of they are 36 (body). Can be 36 more common in the other regions?
            print('Detected MV')
            if save == True:
                writevtk(c, filename[0:len(filename) - 4] + '_cont_mv.vtk')
            cont_mv = c
        if mostcommon == 37:
            print('Detected LAA')
            if save == True:
                writevtk(c, filename[0:len(filename) - 4] + '_cont_laa.vtk')
            cont_laa = c
        if mostcommon == 76:
            print('Detected RSPV')
            if save == True:
                writevtk(c, filename[0:len(filename) - 4] + '_cont_rspv.vtk')
            cont_rspv = c
        if mostcommon == 77:
            print('Detected RIPV')
            if save == True:
                writevtk(c, filename[0:len(filename) - 4] + '_cont_ripv.vtk')
            cont_ripv = c

        if mostcommon == 78:
            print('Detected LSPV')
            if save == True:
                writevtk(c, filename[0:len(filename) - 4] + '_cont_lspv.vtk')
            cont_lspv = c
        if mostcommon == 79:
            print('Detected LIPV')
            
            if save == True:
                writevtk(c, filename[0:len(filename) - 4] + '_cont_lipv.vtk')
            cont_lipv = c
            
    return m_whole, cont_rspv, cont_ripv, cont_lipv, cont_lspv, cont_mv, cont_laa

def build_locators(mesh, m_open, cont_rspv, cont_ripv, cont_lipv, cont_lspv, cont_laa):
    """Build different locators to find corresponding points between different meshes (open/closed, open/contours, etc)"""
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)  # clipped + CLOSED - where the seeds are marked
    locator.BuildLocator()

    locator_open = vtk.vtkPointLocator()
    locator_open.SetDataSet(m_open)
    locator_open.BuildLocator()

    locator_rspv = vtk.vtkPointLocator()
    locator_rspv.SetDataSet(cont_rspv)
    locator_rspv.BuildLocator()

    locator_ripv = vtk.vtkPointLocator()
    locator_ripv.SetDataSet(cont_ripv)
    locator_ripv.BuildLocator()

    locator_lipv = vtk.vtkPointLocator()
    locator_lipv.SetDataSet(cont_lipv)
    locator_lipv.BuildLocator()

    locator_lspv = vtk.vtkPointLocator()
    locator_lspv.SetDataSet(cont_lspv)
    locator_lspv.BuildLocator()

    locator_laa = vtk.vtkPointLocator()
    locator_laa.SetDataSet(cont_laa)
    locator_laa.BuildLocator()
    return locator, locator_open, locator_rspv, locator_ripv, locator_lipv, locator_lspv, locator_laa

def get_mv_contour_ids(cont_mv, locator_open):
    """Obtain ids of the MV contour"""
    edge_cont_ids = get_ordered_cont_ids_based_on_distance(cont_mv)
    mv_cont_ids = np.zeros(edge_cont_ids.size)
    for i in range(mv_cont_ids.shape[0]):
        p = cont_mv.GetPoint(edge_cont_ids[i])
        mv_cont_ids[i] = locator_open.FindClosestPoint(p)
    return mv_cont_ids

def get_ids(line, locator_open):
    """Get ids of points in the open mesh corresponding to the points in the line"""
    npts = line.GetNumberOfPoints()
    ids = np.zeros(npts)
    for i in range(npts):
        p = line.GetPoint(i)
        ids[i] = locator_open.FindClosestPoint(p)
    return ids

def identify_segments_extremes(path1, path2, path3, path4, path5, path6, path7, path_laa1, path_laa2, path_laa3,
                               locator_open, locator_rspv, locator_ripv, locator_lipv, locator_lspv, locator_laa,
                               cont_rspv, cont_ripv, cont_lipv, cont_lspv, cont_laa):
    """Identify ids in the to_be_flat mesh corresponding to the segment extremes: v1d, v1r, ect."""
    # start with segments of PVs because they will modify the rest of segments (we try to have uniform number of points in the 3 segments of the veins)
    # first identify ALL pv segments extremes (v1d, v2u etc.)

    # s1 - Find ids corresponding to v1d and v2u as intersection of rspv (ripv) contour and path1
    dists1 = np.zeros(path1.GetNumberOfPoints())
    dists2 = np.zeros(path1.GetNumberOfPoints())
    for i in range(path1.GetNumberOfPoints()):
        p = path1.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_rspv.GetPoint(locator_rspv.FindClosestPoint(p)))
        dists2[i] = euclideandistance(p, cont_ripv.GetPoint(locator_ripv.FindClosestPoint(p)))
    v1d_in_path1 = np.argmin(dists1)
    v2u_in_path1 = np.argmin(dists2)
    v1d = locator_open.FindClosestPoint(path1.GetPoint(v1d_in_path1))
    v2u = locator_open.FindClosestPoint(path1.GetPoint(v2u_in_path1))

    # s2 - Find 2l and v3r
    dists1 = np.zeros(path2.GetNumberOfPoints())
    dists2 = np.zeros(path2.GetNumberOfPoints())
    for i in range(path2.GetNumberOfPoints()):
        p = path2.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_ripv.GetPoint(locator_ripv.FindClosestPoint(p)))
        dists2[i] = euclideandistance(p, cont_lipv.GetPoint(locator_lipv.FindClosestPoint(p)))
    v2l_in_path2 = np.argmin(dists1)
    v3r_in_path2 = np.argmin(dists2)
    v2l = locator_open.FindClosestPoint(path2.GetPoint(v2l_in_path2))
    v3r = locator_open.FindClosestPoint(path2.GetPoint(v3r_in_path2))

    # s3 - Find v3u and v4d
    dists1 = np.zeros(path3.GetNumberOfPoints())
    dists2 = np.zeros(path3.GetNumberOfPoints())
    for i in range(path3.GetNumberOfPoints()):
        p = path3.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_lipv.GetPoint(locator_lipv.FindClosestPoint(p)))
        dists2[i] = euclideandistance(p, cont_lspv.GetPoint(locator_lspv.FindClosestPoint(p)))
    v3u_in_path3 = np.argmin(dists1)
    v4d_in_path3 = np.argmin(dists2)
    v3u = locator_open.FindClosestPoint(path3.GetPoint(v3u_in_path3))
    v4d = locator_open.FindClosestPoint(path3.GetPoint(v4d_in_path3))

    # s4 - Find v4r and v1l
    dists1 = np.zeros(path4.GetNumberOfPoints())
    dists2 = np.zeros(path4.GetNumberOfPoints())
    for i in range(path4.GetNumberOfPoints()):
        p = path4.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_lspv.GetPoint(locator_lspv.FindClosestPoint(p)))
        dists2[i] = euclideandistance(p, cont_rspv.GetPoint(locator_rspv.FindClosestPoint(p)))
    v4r_in_path4 = np.argmin(dists1)
    v1l_in_path4 = np.argmin(dists2)
    v4r = locator_open.FindClosestPoint(path4.GetPoint(v4r_in_path4))
    v1l = locator_open.FindClosestPoint(path4.GetPoint(v1l_in_path4))

    # Next 4 segments: s5, s6, s7, s8 : FROM pvs (v1r,v2r,v3l,v4l) TO points in the MV
    dists1 = np.zeros(path5.GetNumberOfPoints())
    for i in range(path5.GetNumberOfPoints()):
        p = path5.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_rspv.GetPoint(locator_rspv.FindClosestPoint(p)))
    v1r_in_path5 = np.argmin(dists1)
    v1r = locator_open.FindClosestPoint(path5.GetPoint(v1r_in_path5))

    # s6
    dists1 = np.zeros(path6.GetNumberOfPoints())
    for i in range(path6.GetNumberOfPoints()):
        p = path6.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_ripv.GetPoint(locator_ripv.FindClosestPoint(p)))
    v2r_in_path6 = np.argmin(dists1)
    v2r = locator_open.FindClosestPoint(path6.GetPoint(v2r_in_path6))

    # s7
    dists1 = np.zeros(path7.GetNumberOfPoints())
    for i in range(path7.GetNumberOfPoints()):
        p = path7.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_lipv.GetPoint(locator_lipv.FindClosestPoint(p)))
    v3l_in_path7 = np.argmin(dists1)
    v3l = locator_open.FindClosestPoint(path7.GetPoint(v3l_in_path7))

    # S8a -> segment from v4 (lspv) to LAA
    dists1 = np.zeros(path_laa1.GetNumberOfPoints())
    dists2 = np.zeros(path_laa1.GetNumberOfPoints())
    for i in range(path_laa1.GetNumberOfPoints()):
        p = path_laa1.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_lspv.GetPoint(locator_lspv.FindClosestPoint(p)))
        dists2[i] = euclideandistance(p, cont_laa.GetPoint(locator_laa.FindClosestPoint(p)))
    v4u_in_pathlaa1 = np.argmin(dists1)
    vlaad_in_pathlaa1 = np.argmin(dists2)
    v4u = locator_open.FindClosestPoint(path_laa1.GetPoint(v4u_in_pathlaa1))
    vlaad = locator_open.FindClosestPoint(path_laa1.GetPoint(vlaad_in_pathlaa1))

    # S8b -> segment from LAA to V8 (MV)
    dists1 = np.zeros(path_laa2.GetNumberOfPoints())
    for i in range(path_laa2.GetNumberOfPoints()):
        p = path_laa2.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_laa.GetPoint(locator_laa.FindClosestPoint(p)))
    vlaau_in_pathlaa2 = np.argmin(dists1)
    vlaau = locator_open.FindClosestPoint(path_laa2.GetPoint(vlaau_in_pathlaa2))

    # aux point vlaar (connecting laa and rspv - auxiliary to know laa contour direction)
    dists1 = np.zeros(path_laa3.GetNumberOfPoints())
    dists2 = np.zeros(path_laa3.GetNumberOfPoints())
    for i in range(path_laa3.GetNumberOfPoints()):
        p = path_laa3.GetPoint(i)
        dists1[i] = euclideandistance(p, cont_rspv.GetPoint(locator_rspv.FindClosestPoint(p)))
        dists2[i] = euclideandistance(p, cont_laa.GetPoint(locator_laa.FindClosestPoint(p)))
        
    v1u_in_pathlaa3 = np.argmin(dists1)
    vlaar_in_pathlaa3 = np.argmin(dists2)
    v1u = locator_open.FindClosestPoint(path_laa3.GetPoint(v1u_in_pathlaa3))
    vlaar = locator_open.FindClosestPoint(path_laa3.GetPoint(vlaar_in_pathlaa3))
    return v1r, v1d, v1l, v1u, v2u, v2r, v2l, v3u, v3r, v3l, v4r, v4u, v4d, vlaad, vlaau, vlaar

def get_rspv_segments_ids(cont_rspv, locator_open, v1l, v1d, v1r,v1u, propn_rspv_s1, propn_rspv_s2, propn_rspv_s3, propn_rspv_s4):
    """ Return 3 arrays with ids of each of the 3 segments in rspv contour.
        Return also the modified (to have proportional number of points in the segments) extreme ids"""
    
    edge_cont_rspv = get_ordered_cont_ids_based_on_distance(cont_rspv)
    rspv_cont_ids = np.zeros(edge_cont_rspv.size)
    for i in range(rspv_cont_ids.shape[0]):
        p = cont_rspv.GetPoint(edge_cont_rspv[i])
        rspv_cont_ids[i] = locator_open.FindClosestPoint(p)
    pos_v1d = int(np.where(rspv_cont_ids == v1d)[0])
    
    rspv_ids = np.append(rspv_cont_ids[pos_v1d:rspv_cont_ids.size], rspv_cont_ids[0:pos_v1d])
    pos_v1r = int(np.where(rspv_ids == v1r)[0])
    pos_v1l = int(np.where(rspv_ids == v1l)[0])
    if pos_v1r < pos_v1l:   # flip
        aux = np.zeros(rspv_ids.size)
        for i in range(rspv_ids.size):
            aux[rspv_ids.size - 1 - i] = rspv_ids[i]
        # maintain the v1l as the first one (after the flip is the last one)
        flipped = np.append(aux[aux.size - 1], aux[0:aux.size - 1])
        rspv_ids = flipped.astype(int)
    print("Positions of v1l, v1d, v1r, v1u", int(np.where(rspv_ids == v1l)[0]),
          int(np.where(rspv_ids == v1d)[0]), int(np.where(rspv_ids == v1r)[0]), int(np.where(rspv_ids == v1u)[0]))
    rspv_s1 = rspv_ids[0:int(np.where(rspv_ids == v1l)[0])]
    rspv_s2 = rspv_ids[int(np.where(rspv_ids == v1l)[0]): int(np.where(rspv_ids == v1u)[0])]
    rspv_s3 = rspv_ids[int(np.where(rspv_ids == v1u)[0]): int(np.where(rspv_ids == v1r)[0])]
    rspv_s4 = rspv_ids[int(np.where(rspv_ids == v1r)[0]): rspv_ids.size]

    s1_prop_length = round(propn_rspv_s1 * len(rspv_ids))
    s2_prop_length = round(propn_rspv_s2 * len(rspv_ids))
    s3_prop_length = round(propn_rspv_s3 * len(rspv_ids))
    v1d_prop = v1d   # stays the same, reference
    rspv_s1_offset = round((s1_prop_length - rspv_s1.size)/2)        # If negative, I'll shorten s1 in that case
    v1l_prop = rspv_ids[int(rspv_s1.size + rspv_s1_offset)]
    rspv_s1_prop = rspv_ids[0:int(rspv_s1.size + rspv_s1_offset)]

    new_s2_size = rspv_s2.size - rspv_s1_offset   # initial minus points now given to s1
    rspv_s2_offset = np.floor((s2_prop_length - new_s2_size)/2)    # I will add an offset of half the difference. Floor, otherwise s3 is always shorter since it is the remaining part
    v1u_prop = rspv_ids[int(rspv_s1_prop.size + new_s2_size + rspv_s2_offset)]
    rspv_s2_prop = rspv_ids[int(rspv_s1.size + rspv_s1_offset):int(rspv_s1.size + rspv_s1_offset + new_s2_size + rspv_s2_offset)]
    
    new_s3_size = rspv_s3.size - rspv_s2_offset   # initial minus points now given to s1
    rspv_s3_offset = np.floor((s3_prop_length - new_s3_size)/2)
    print(int(rspv_s1_prop.size + rspv_s2_prop.size + new_s3_size + rspv_s3_offset))
    v1r_prop = rspv_ids[int(rspv_s1_prop.size + rspv_s2_prop.size + new_s3_size + rspv_s3_offset)]
    rspv_s3_prop = rspv_ids[int(rspv_s1.size + rspv_s1_offset + new_s2_size + rspv_s2_offset): int(rspv_s1_prop.size + rspv_s2_prop.size + new_s3_size + rspv_s3_offset)]
    rspv_s4_prop = rspv_ids[int(rspv_s1_prop.size + rspv_s2_prop.size + new_s3_size + rspv_s3_offset): rspv_ids.size]

    print("Positions of v1l, v1d, v1r, v1u", int(np.where(rspv_ids == v1l_prop)[0]),
          int(np.where(rspv_ids == v1d_prop)[0]), int(np.where(rspv_ids == v1r_prop)[0]), int(np.where(rspv_ids == v1u_prop)[0]))
    print('RSPV original lengths', rspv_s1.size, rspv_s2.size, rspv_s3.size, rspv_s4.size)
    print('Proportional lengths', rspv_s1_prop.size, rspv_s2_prop.size, rspv_s3_prop.size, rspv_s4_prop.size)
    return rspv_ids, rspv_s1_prop, rspv_s2_prop, rspv_s3_prop,rspv_s4_prop, v1l_prop, v1d_prop, v1r_prop, v1u_prop

def get_ripv_segments_ids(cont_ripv, locator_open, v2l, v2r, v2u, propn_ripv_s1, propn_ripv_s2, propn_ripv_s3):
    """ Return 3 arrays with ids of each of the 3 segments in ripv contour.
        Return also the modified (to have proportional number of points in the segments) extreme ids"""
    edge_cont_ripv = get_ordered_cont_ids_based_on_distance(cont_ripv)
    ripv_cont_ids = np.zeros(edge_cont_ripv.size)
    for i in range(ripv_cont_ids.shape[0]):
        p = cont_ripv.GetPoint(edge_cont_ripv[i])
        ripv_cont_ids[i] = locator_open.FindClosestPoint(p)
    pos_v2l = int(np.where(ripv_cont_ids == v2l)[0])
    ripv_ids = np.append(ripv_cont_ids[pos_v2l:ripv_cont_ids.size], ripv_cont_ids[0:pos_v2l])
    pos_v2r = int(np.where(ripv_ids == v2r)[0])
    pos_v2u = int(np.where(ripv_ids == v2u)[0])
    if pos_v2u < pos_v2r:  # flip
        aux = np.zeros(ripv_ids.size)
        for i in range(ripv_ids.size):
            aux[ripv_ids.size - 1 - i] = ripv_ids[i]
        flipped = np.append(aux[aux.size - 1], aux[0:aux.size - 1])
        ripv_ids = flipped.astype(int)
    ripv_s1 = ripv_ids[0:int(np.where(ripv_ids == v2r)[0])]
    ripv_s2 = ripv_ids[int(np.where(ripv_ids == v2r)[0]): int(np.where(ripv_ids == v2u)[0])]
    ripv_s3 = ripv_ids[int(np.where(ripv_ids == v2u)[0]): ripv_ids.size]

    s1_prop_length = round(propn_ripv_s1 * len(ripv_ids))
    s2_prop_length = round(propn_ripv_s2 * len(ripv_ids))
    s3_prop_length = round(propn_ripv_s3 * len(ripv_ids))
    v2l_prop = v2l   # stays the same, reference
    ripv_s1_offset = round((s1_prop_length - ripv_s1.size)/2)
    v2r_prop = ripv_ids[int(ripv_s1.size + ripv_s1_offset)]
    ripv_s1_prop = ripv_ids[0:int(ripv_s1.size + ripv_s1_offset)]
    new_s2_size = ripv_s2.size - ripv_s1_offset
    ripv_s2_offset = np.floor((s2_prop_length - new_s2_size)/2)
    v2u_prop = ripv_ids[int(ripv_s1_prop.size + new_s2_size + ripv_s2_offset)]
    ripv_s2_prop = ripv_ids[int(ripv_s1.size + ripv_s1_offset):int(ripv_s1.size + ripv_s1_offset + new_s2_size + ripv_s2_offset)]
    ripv_s3_prop = ripv_ids[int(ripv_s1.size + ripv_s1_offset + new_s2_size + ripv_s2_offset): ripv_ids.size]
    print('RIPV original lengths', ripv_s1.size, ripv_s2.size, ripv_s3.size)
    print('Proportional lengths', ripv_s1_prop.size, ripv_s2_prop.size, ripv_s3_prop.size)
    return ripv_ids, ripv_s1_prop, ripv_s2_prop, ripv_s3_prop, v2l_prop, v2r_prop, v2u_prop

def get_lipv_segments_ids(cont_lipv, locator_open, v3r, v3u, v3l, propn_lipv_s1, propn_lipv_s2, propn_lipv_s3):
    """ Return 3 arrays with ids of each of the 3 segments in lipv contour.
        Return also the modified (to have proportional number of points in the segments) extreme ids"""
    edge_cont_lipv = get_ordered_cont_ids_based_on_distance(cont_lipv)
    lipv_cont_ids = np.zeros(edge_cont_lipv.size)
    for i in range(lipv_cont_ids.shape[0]):
        p = cont_lipv.GetPoint(edge_cont_lipv[i])
        lipv_cont_ids[i] = locator_open.FindClosestPoint(p)
    pos_v3u = int(np.where(lipv_cont_ids == v3u)[0])
    lipv_ids = np.append(lipv_cont_ids[pos_v3u:lipv_cont_ids.size], lipv_cont_ids[0:pos_v3u])
    pos_v3l = int(np.where(lipv_ids == v3l)[0])
    pos_v3r = int(np.where(lipv_ids == v3r)[0])
    if pos_v3r < pos_v3l:  # flip
        aux = np.zeros(lipv_ids.size)
        for i in range(lipv_ids.size):
            aux[lipv_ids.size - 1 - i] = lipv_ids[i]
        flipped = np.append(aux[aux.size - 1], aux[0:aux.size - 1])
        lipv_ids = flipped.astype(int)
    lipv_s1 = lipv_ids[0:int(np.where(lipv_ids == v3l)[0])]
    lipv_s2 = lipv_ids[int(np.where(lipv_ids == v3l)[0]): int(np.where(lipv_ids == v3r)[0])]
    lipv_s3 = lipv_ids[int(np.where(lipv_ids == v3r)[0]): lipv_ids.size]
    
    s1_prop_length = round(propn_lipv_s1 * len(lipv_ids))
    s2_prop_length = round(propn_lipv_s2 * len(lipv_ids))
    s3_prop_length = round(propn_lipv_s3 * len(lipv_ids))
    v3u_prop = v3u   # stays the same, reference
    lipv_s1_offset = round((s1_prop_length - lipv_s1.size)/2)
    v3l_prop = lipv_ids[int(lipv_s1.size + lipv_s1_offset)]
    lipv_s1_prop = lipv_ids[0:int(lipv_s1.size + lipv_s1_offset)]
    new_s2_size = lipv_s2.size - lipv_s1_offset
    lipv_s2_offset = np.floor((s2_prop_length - new_s2_size)/2)
    v3r_prop = lipv_ids[int(lipv_s1_prop.size + new_s2_size + lipv_s2_offset)]
    lipv_s2_prop = lipv_ids[int(lipv_s1.size + lipv_s1_offset):int(lipv_s1.size + lipv_s1_offset + new_s2_size + lipv_s2_offset)]
    lipv_s3_prop = lipv_ids[int(lipv_s1.size + lipv_s1_offset + new_s2_size + lipv_s2_offset): lipv_ids.size]
    print('LIPV original lengths', lipv_s1.size, lipv_s2.size, lipv_s3.size)
    print('Proportional lengths', lipv_s1_prop.size, lipv_s2_prop.size, lipv_s3_prop.size)
    print('LIPV v3r, v3u, v3l', v3r_prop, v3u_prop, v3l_prop)
    print("Positions of v3r, v3u, v3l", int(np.where(lipv_ids == v3r_prop)[0]),
          int(np.where(lipv_ids == v3u_prop)[0]), int(np.where(lipv_ids == v3l_prop)[0]))
    return lipv_ids, lipv_s1_prop, lipv_s2_prop, lipv_s3_prop, v3r_prop, v3u_prop, v3l_prop

def get_lspv_segments_ids(cont_lspv, locator_open, v4r, v4u, v4d, propn_lspv_s1, propn_lspv_s2, propn_lspv_s3):
    """ Return 3 arrays with ids of each of the 3 segments in lspv contour.
        Return also the modified (to have proportional number of points in the segments) extreme ids"""
    edge_cont_lspv = get_ordered_cont_ids_based_on_distance(cont_lspv)
    lspv_cont_ids = np.zeros(edge_cont_lspv.size)
    for i in range(lspv_cont_ids.shape[0]):
        p = cont_lspv.GetPoint(edge_cont_lspv[i])
        lspv_cont_ids[i] = locator_open.FindClosestPoint(p)
    pos_v4r = int(np.where(lspv_cont_ids == v4r)[0])
    lspv_ids = np.append(lspv_cont_ids[pos_v4r:lspv_cont_ids.size], lspv_cont_ids[0:pos_v4r])
    pos_v4u = int(np.where(lspv_ids == v4u)[0])
    pos_v4d = int(np.where(lspv_ids == v4d)[0])
    if pos_v4d < pos_v4u:   # flip
        aux = np.zeros(lspv_ids.size)
        for i in range(lspv_ids.size):
            aux[lspv_ids.size - 1 - i] = lspv_ids[i]
        flipped = np.append(aux[aux.size - 1], aux[0:aux.size - 1])
        lspv_ids = flipped.astype(int)
    lspv_s1 = lspv_ids[0:int(np.where(lspv_ids == v4u)[0])]
    lspv_s2 = lspv_ids[int(np.where(lspv_ids == v4u)[0]): int(np.where(lspv_ids == v4d)[0])]
    lspv_s3 = lspv_ids[int(np.where(lspv_ids == v4d)[0]): lspv_ids.size]

    s1_prop_length = round(propn_lspv_s1*len(lspv_ids))
    s2_prop_length = round(propn_lspv_s2*len(lspv_ids))
    s3_prop_length = round(propn_lspv_s3*len(lspv_ids))
    v4r_prop = v4r   # stays the same, reference
    lspv_s1_offset = round((s1_prop_length - lspv_s1.size)/2)
    v4u_prop = lspv_ids[int(lspv_s1.size + lspv_s1_offset)]
    lspv_s1_prop = lspv_ids[0:int(lspv_s1.size + lspv_s1_offset)]
    new_s2_size = lspv_s2.size - lspv_s1_offset
    lspv_s2_offset = np.floor((s2_prop_length - new_s2_size)/2)
    v4d_prop = lspv_ids[int(lspv_s1_prop.size + new_s2_size + lspv_s2_offset)]
    lspv_s2_prop = lspv_ids[int(lspv_s1.size + lspv_s1_offset):int(lspv_s1.size + lspv_s1_offset + new_s2_size + lspv_s2_offset)]
    lspv_s3_prop = lspv_ids[int(lspv_s1.size + lspv_s1_offset + new_s2_size + lspv_s2_offset): lspv_ids.size]
    print('LSPV Original lengths', lspv_s1.size, lspv_s2.size, lspv_s3.size)
    print('Proportional lengths', lspv_s1_prop.size, lspv_s2_prop.size, lspv_s3_prop.size)
    return lspv_ids, lspv_s1_prop, lspv_s2_prop, lspv_s3_prop, v4r_prop, v4u_prop, v4d_prop

def get_laa_segments_ids(cont_laa, locator_open, vlaau, vlaad, vlaar, propn_laa_s1, propn_laa_s2, propn_laa_s3):
    edge_cont_laa = get_ordered_cont_ids_based_on_distance(cont_laa)
    laa_cont_ids = np.zeros(edge_cont_laa.size)
    for i in range(laa_cont_ids.shape[0]):
        p = cont_laa.GetPoint(edge_cont_laa[i])
        laa_cont_ids[i] = locator_open.FindClosestPoint(p)
    pos_vlaar = int(np.where(laa_cont_ids == vlaar)[0])  # intersection of laa contour and path 8a (from lspv to laa)
    laa_ids = np.append(laa_cont_ids[pos_vlaar:laa_cont_ids.size], laa_cont_ids[0:pos_vlaar])

    pos_vlaad = int(np.where(laa_ids == vlaad)[0])
    pos_vlaau = int(np.where(laa_ids == vlaau)[0])
    if pos_vlaad < pos_vlaau:  # flip
        aux = np.zeros(laa_ids.size)
        for i in range(laa_ids.size):
            aux[laa_ids.size - 1 - i] = laa_ids[i]
        flipped = np.append(aux[aux.size - 1], aux[0:aux.size - 1])
        laa_ids = flipped.astype(int)
    
    laa_s1 = laa_ids[0:int(np.where(laa_ids == vlaau)[0])]
    laa_s2 = laa_ids[int(np.where(laa_ids == vlaau)[0]): int(np.where(laa_ids == vlaad)[0])]
    laa_s3 = laa_ids[int(np.where(laa_ids == vlaad)[0]): laa_ids.size]

    s1_prop_length = round(propn_laa_s1*len(laa_ids))
    s2_prop_length = round(propn_laa_s2*len(laa_ids))
    s3_prop_length = round(propn_laa_s3*len(laa_ids))
    vlaar_prop = vlaar   # stays the same, reference
    laa_s1_offset = round((s1_prop_length - laa_s1.size)/2)
    vlaau_prop = laa_ids[int(laa_s1.size + laa_s1_offset)]
    laa_s1_prop = laa_ids[0:int(laa_s1.size + laa_s1_offset)]
    new_s2_size = laa_s2.size - laa_s1_offset
    laa_s2_offset = np.floor((s2_prop_length - new_s2_size)/2)
    vlaad_prop = laa_ids[int(laa_s1_prop.size + new_s2_size + laa_s2_offset)]
    laa_s2_prop = laa_ids[int(laa_s1.size + laa_s1_offset):int(laa_s1.size + laa_s1_offset + new_s2_size + laa_s2_offset)]
    laa_s3_prop = laa_ids[int(laa_s1.size + laa_s1_offset + new_s2_size + laa_s2_offset): laa_ids.size]
    print("Positions of vlaau, vlaad, vlaar in laa_ids", int(np.where(laa_ids == vlaau_prop)[0]),
          int(np.where(laa_ids == vlaad_prop)[0]), int(np.where(laa_ids == vlaar_prop)[0]))
    print('LAA Original lengths', laa_s1.size, laa_s2.size, laa_s3.size)
    print('Proportional lengths', laa_s1_prop.size, laa_s2_prop.size, laa_s3_prop.size)
    return laa_ids, laa_s1_prop, laa_s2_prop, laa_s3_prop, vlaau_prop, vlaad_prop, vlaar_prop

def get_mv_segments_ids(cont_mv, locator_open,
                        vm1, vm2, vm3, vm4,
                        propn_mv_s1, propn_mv_s2, propn_mv_s3, propn_mv_s4):
    """Return 4 arrays with ids of each of the 4 segments in MV contour.
       Return also the modified (to have proportional number of points in the segments) extreme ids."""
    # Order contour points consistently
    edge_cont_mv = get_ordered_cont_ids_based_on_distance(cont_mv)
    mv_cont_ids = np.zeros(edge_cont_mv.size)
    for i in range(mv_cont_ids.shape[0]):
        p = cont_mv.GetPoint(edge_cont_mv[i])
        mv_cont_ids[i] = locator_open.FindClosestPoint(p)

    # 2Rotate contour so vm1 is the start
    pos_vm4 = int(np.where(mv_cont_ids == vm4)[0])
    mv_ids = np.append(mv_cont_ids[pos_vm4:mv_cont_ids.size], mv_cont_ids[0:pos_vm4])

    # Locate the other key points
    pos_vm2 = int(np.where(mv_ids == vm2)[0])
    pos_vm3 = int(np.where(mv_ids == vm3)[0])
    pos_vm1 = int(np.where(mv_ids == vm1)[0])

    # Ensure correct orientation (optional flip)
    if pos_vm1 < pos_vm3:
        mv_ids = mv_ids.astype(int)
    else:
        aux = np.zeros(mv_ids.size)
        for i in range(mv_ids.size):
            aux[mv_ids.size-1-i] = mv_ids[i]
        flipped = np.append(aux[aux.size-1], aux[0:aux.size-1])
        mv_ids = flipped.astype(int)

    # Split into 4 original segments
    mv_s4 = mv_ids[0:int(np.where(mv_ids == vm1)[0])]
    mv_s1 = mv_ids[int(np.where(mv_ids == vm1)[0]): int(np.where(mv_ids == vm2)[0])]
    mv_s2 = mv_ids[int(np.where(mv_ids == vm2)[0]): int(np.where(mv_ids == vm3)[0])]
    mv_s3 = mv_ids[int(np.where(mv_ids == vm3)[0]): mv_ids.size]

    # Adjust to have proportional lengths (INTERMEDIATE SOLUTION)
    s1_prop_length = round(propn_mv_s1 * len(mv_ids))
    s2_prop_length = round(propn_mv_s2 * len(mv_ids))
    s3_prop_length = round(propn_mv_s3 * len(mv_ids))
    s4_prop_length = round(propn_mv_s4 * len(mv_ids))
    print('MV segment desired lengths:', s1_prop_length, s2_prop_length, s3_prop_length, s4_prop_length)
    vm4_prop = vm4  # reference
    mv_s4_offset = round((s4_prop_length - mv_s4.size) / 2)
    vm1_prop = mv_ids[int(mv_s4.size + mv_s4_offset)]
    mv_s4_prop = mv_ids[0:int(mv_s4.size + mv_s4_offset)]

    new_s1_size = mv_s1.size - mv_s4_offset
    mv_s1_offset = np.floor((s1_prop_length - new_s1_size) / 2)
    vm2_prop = mv_ids[int(mv_s4_prop.size + new_s1_size + mv_s1_offset)]
    mv_s1_prop = mv_ids[int(mv_s4.size + mv_s4_offset):int(mv_s4.size + mv_s4_offset + new_s1_size + mv_s1_offset)]

    new_s2_size = mv_s2.size - mv_s1_offset
    mv_s2_offset = np.floor((s2_prop_length - new_s2_size) / 2)
    vm3_prop = mv_ids[int(mv_s4_prop.size + mv_s1_prop.size + new_s2_size + mv_s2_offset)]
    mv_s2_prop = mv_ids[int(mv_s4_prop.size + mv_s1_prop.size):int(mv_s4_prop.size + mv_s1_prop.size + new_s2_size + mv_s2_offset)]

    mv_s3_prop = mv_ids[int(mv_s4_prop.size + mv_s1_prop.size + mv_s2_prop.size): mv_ids.size]
    print('MV Original lengths', mv_s1.size, mv_s2.size, mv_s3.size, mv_s4.size)
    print('Proportional lengths', mv_s1_prop.size, mv_s2_prop.size, mv_s3_prop.size, mv_s4_prop.size)

    vm4_prop = vm4  # reference
    vm1_prop = mv_ids[s4_prop_length]
    mv_s4_prop = mv_ids[0:int(s4_prop_length)]

    vm2_prop = mv_ids[int(mv_s4_prop.size + s1_prop_length)]
    mv_s1_prop = mv_ids[int(mv_s4_prop.size):int(mv_s4_prop.size + s1_prop_length)]

    vm3_prop = mv_ids[int(mv_s4_prop.size + mv_s1_prop.size + s2_prop_length)]
    mv_s2_prop = mv_ids[int(mv_s4_prop.size + mv_s1_prop.size):int(mv_s4_prop.size + mv_s1_prop.size + s2_prop_length)]

    mv_s3_prop = mv_ids[int(mv_s4_prop.size + mv_s1_prop.size + mv_s2_prop.size): mv_ids.size]
    print('MV original segment sizes:', mv_s1.size, mv_s2.size, mv_s3.size, mv_s4.size)
    print('MV proportional sizes:', mv_s1_prop.size, mv_s2_prop.size, mv_s3_prop.size, mv_s4_prop.size)

    return mv_ids, mv_s1_prop, mv_s2_prop, mv_s3_prop, mv_s4_prop, vm1_prop, vm2_prop, vm3_prop, vm4_prop

def paths_intersect(path1_ids,path2_ids):

    path1_set = set(path1_ids)
    path2_set = set(path2_ids)
    # If the path shares any vertex with the contour, it's considered intersecting
    return len(path1_set.intersection(path2_set)) > 0

def get_segment_ids_in_to_be_flat_mesh(path, locator, intersect_end, intersect_beginning):
    s = np.zeros(path.GetNumberOfPoints())
    for i in range(path.GetNumberOfPoints()):
        p = path.GetPoint(i)
        s[i] = int(locator.FindClosestPoint(p))
    intersect_wlast = np.intersect1d(s, intersect_end)   # find repeated values (s1 merges with rspv contour)
    nlasts_to_delete = len(intersect_wlast)
    index1 = np.arange(len(s) - nlasts_to_delete, len(s))
    final_s = np.delete(s, index1)

    intersect_wfirst = np.intersect1d(final_s, intersect_beginning)
    nfirst_to_delete = len(intersect_wfirst)
    index2 = np.arange(0, nfirst_to_delete)
    s = np.delete(final_s, index2)
    return s

def find_location_of_repeated_ids(overlap, s1, s2, s3, s4, s5, s6, s7, s8a, s8b, s8c, mv_s1_prop, mv_s2_prop, mv_s3_prop, mv_s4_prop, 
    rspv_s1_prop, rspv_s2_prop, rspv_s3_prop, rspv_s4_prop, ripv_s1_prop, ripv_s2_prop, ripv_s3_prop, lipv_s1_prop, lipv_s2_prop, lipv_s3_prop, 
    lspv_s1_prop, lspv_s2_prop, lspv_s3_prop, laa_s1, laa_s2, laa_s3):
    for oid in overlap:
        print("ID:", oid)

        # -------- Constraint segments --------
        if oid in s1: print("  in constraint: s1")
        if oid in s2: print("  in constraint: s2")
        if oid in s3: print("  in constraint: s3")
        if oid in s4: print("  in constraint: s4")
        if oid in s5: print("  in constraint: s5")
        if oid in s6: print("  in constraint: s6")
        if oid in s7: print("  in constraint: s7")
        if oid in s8a: print("  in constraint: s8a")
        if oid in s8b: print("  in constraint: s8b")
        if oid in s8c: print("  in constraint: s8c")

        # -------- Mitral valve (MV) --------
        if oid in mv_s1_prop: print("  in contour: mv_s1")
        if oid in mv_s2_prop: print("  in contour: mv_s2")
        if oid in mv_s3_prop: print("  in contour: mv_s3")
        if oid in mv_s4_prop: print("  in contour: mv_s4")

        # -------- RSPV --------
        if oid in rspv_s1_prop: print("  in contour: rspv_s1")
        if oid in rspv_s2_prop: print("  in contour: rspv_s2")
        if oid in rspv_s3_prop: print("  in contour: rspv_s3")
        if oid in rspv_s4_prop: print("  in contour: rspv_s4")

        # -------- RIPV --------
        if oid in ripv_s1_prop: print("  in contour: ripv_s1")
        if oid in ripv_s2_prop: print("  in contour: ripv_s2")
        if oid in ripv_s3_prop: print("  in contour: ripv_s3")

        # -------- LIPV --------
        if oid in lipv_s1_prop: print("  in contour: lipv_s1")
        if oid in lipv_s2_prop: print("  in contour: lipv_s2")
        if oid in lipv_s3_prop: print("  in contour: lipv_s3")

        # -------- LSPV --------
        if oid in lspv_s1_prop: print("  in contour: lspv_s1")
        if oid in lspv_s2_prop: print("  in contour: lspv_s2")
        if oid in lspv_s3_prop: print("  in contour: lspv_s3")

        # -------- LAA --------
        if oid in laa_s1: print("  in contour: laa_s1")
        if oid in laa_s2: print("  in contour: laa_s2")
        if oid in laa_s3: print("  in contour: laa_s3")

def define_boundary_positions(rdisk, rhole_rspv, rhole_ripv, rhole_lipv, rhole_lspv, rhole_laa, xhole_center, yhole_center, laa_hole_center_x, laa_hole_center_y,
                              segment_lengths, t_v5_1, t_v5_2, t_v6, t_v7, t_v8, args):
    """Define BOUNDARY target (x0,y0) coordinates given template parameters (hole radii and positions) and number of points of segments"""
    p_bound = np.sum(segment_lengths)
    x0_bound = np.zeros(int(p_bound))
    y0_bound = np.zeros(int(p_bound))
    # start with BOUNDARY (disk contour) 4 segments of the mv <-> contour of the disk
    # s9: left
    ind1 = 0
    ind2 = segment_lengths[5, 0]
    t = np.linspace(-(2*np.pi - t_v6), t_v5_1, segment_lengths[5, 0] +1, endpoint=True)   # +1 because later I will exclude the last point
    # flip to have clock wise direction in the angle
    aux = np.zeros(t.size)
    for i in range(t.size):
        aux[t.size-1-i] = t[i]
    t = aux
    final_t = t[0:len(t)-1]  # exclude extreme, only one, last
    x0_bound[ind1: ind2] = np.cos(final_t) * rdisk
    y0_bound[ind1: ind2] = np.sin(final_t) * rdisk

    # s10: bottom
    ind1 = ind2
    ind2 = ind2 + segment_lengths[5, 1]
    t = np.linspace(t_v7, t_v6, segment_lengths[5, 1]+1, endpoint=True)
    # flip to have clock wise direction in the angle
    aux = np.zeros(t.size)
    for i in range(t.size):
        aux[t.size-1-i] = t[i]
    t = aux
    final_t = t[0:len(t)-1]  # exclude extreme, only one, last
    x0_bound[ind1: ind2] = np.cos(final_t) * rdisk
    y0_bound[ind1: ind2] = np.sin(final_t) * rdisk

    # s11: left - from v7 to v8
    ind1 = ind2
    ind2 = ind2 + segment_lengths[5, 2]
    t = np.linspace(t_v8, t_v7, segment_lengths[5, 2]+1, endpoint=True)
    # flip to have clock wise direction in the angle
    aux = np.zeros(t.size)
    for i in range(t.size):
        aux[t.size-1-i] = t[i]
    t = aux
    final_t = t[0:len(t)-1]  # exclude extreme, only one, last
    x0_bound[ind1: ind2] = np.cos(final_t) * rdisk
    y0_bound[ind1: ind2] = np.sin(final_t) * rdisk

    # s12: top
    ind1 = ind2
    ind2 = ind2 + segment_lengths[5, 3]
    t = np.linspace(t_v5_1, t_v8, segment_lengths[5, 3]+1, endpoint=True)
    # flip to have clock wise direction in the angle
    aux = np.zeros(t.size)
    for i in range(t.size):
        aux[t.size-1-i] = t[i]
    t = aux
    final_t = t[0:len(t)-1]  # exclude extreme, only one, last
    x0_bound[ind1: ind2] = np.cos(final_t) * rdisk
    y0_bound[ind1: ind2] = np.sin(final_t) * rdisk

    # PV HOLES
    # RSPV, starts in pi
    # rspv_s1
    ind1 = ind2
    ind2 = ind2 + segment_lengths[0,0]
    t = np.linspace(3*np.pi/2, np.pi ,  segment_lengths[0, 0]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_rspv + xhole_center[0]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_rspv + yhole_center[0]

    # rspv_s2
    ind1 = ind2
    ind2 = ind2 + segment_lengths[0,1]
    t = np.linspace(np.pi , np.pi - t_v5_2,  segment_lengths[0, 1]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_rspv + xhole_center[0]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_rspv + yhole_center[0]
    
    # rspv_s3
    ind1 = ind2
    ind2 = ind2 + segment_lengths[0,2]
    t = np.linspace(np.pi - t_v5_2, t_v5_1, segment_lengths[0, 2]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_rspv + xhole_center[0]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_rspv + yhole_center[0]

    # rspv_s4
    ind1 = ind2
    ind2 = ind2 + segment_lengths[0, 3]
    t = np.linspace(2*np.pi + t_v5_1,  3*np.pi/2, segment_lengths[0, 3]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_rspv + xhole_center[0]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_rspv + yhole_center[0]

    # RIPV, starts in pi
    ind1 = ind2
    ind2 = ind2 + segment_lengths[1,0]
    t = np.linspace(np.pi, t_v6, segment_lengths[1, 0]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_ripv + xhole_center[1]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_ripv + yhole_center[1]
    # ripv_s2
    ind1 = ind2
    ind2 = ind2 + segment_lengths[1,1]
    t = np.linspace(t_v6, 2*np.pi + np.pi/2, segment_lengths[1, 1]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_ripv + xhole_center[1]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_ripv + yhole_center[1]
    # ripv_s3
    ind1 = ind2
    ind2 = ind2 + segment_lengths[1,2]
    t = np.linspace(np.pi/2, np.pi, segment_lengths[1, 2]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_ripv + xhole_center[1]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_ripv + yhole_center[1]

    # LIPV, starts in 0
    # lipv_s1
    ind1 = ind2
    ind2 = ind2 + segment_lengths[2, 0]
    t = np.linspace(np.pi/2, t_v7, segment_lengths[2, 0]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_lipv + xhole_center[2]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_lipv + yhole_center[2]

    # lipv_s2
    ind1 = ind2
    ind2 = ind2 + segment_lengths[2, 1]
    t = np.linspace(t_v7, 2*np.pi, segment_lengths[2, 1]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_lipv + xhole_center[2]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_lipv + yhole_center[2]
    
    # lipv_s3
    ind1 = ind2
    ind2 = ind2 + segment_lengths[2, 2]
    t = np.linspace(0, np.pi/2, segment_lengths[2, 2]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_lipv + xhole_center[2]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_lipv + yhole_center[2]

    # LSPV, starts in 0
    # lspv_s1
    ind1 = ind2
    ind2 = ind2 + segment_lengths[3, 0]
    t = np.linspace(0, np.pi/2, segment_lengths[3, 0]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_lspv + xhole_center[3]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_lspv + yhole_center[3]
    # lspv_s2
    ind1 = ind2
    ind2 = ind2 + segment_lengths[3, 1]
    t = np.linspace(np.pi/2, 3*np.pi/2, segment_lengths[3, 1]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_lspv + xhole_center[3]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_lspv + yhole_center[3]
    # lspv_s3
    ind1 = ind2
    ind2 = ind2 + segment_lengths[3, 2]
    t = np.linspace(3*np.pi/2, 2*np.pi, segment_lengths[3, 2]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_lspv + xhole_center[3]
    y0_bound[ind1: ind2] = np.sin(t) * rhole_lspv + yhole_center[3]

    # LAA hole, circumf
    # laa s1
    ind1 = ind2
    ind2 = ind2 + segment_lengths[4, 0]
    t = np.linspace(0, t_v8, segment_lengths[4, 0]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_laa + laa_hole_center_x
    y0_bound[ind1: ind2] = np.sin(t) * rhole_laa + laa_hole_center_y
    # laa s2
    ind1 = ind2
    ind2 = ind2 + segment_lengths[4, 1]
    t = np.linspace(t_v8, 3*np.pi/2, segment_lengths[4, 1]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_laa + laa_hole_center_x
    y0_bound[ind1: ind2] = np.sin(t) * rhole_laa + laa_hole_center_y
    # laa s3
    ind1 = ind2
    ind2 = ind2 + segment_lengths[4, 2]
    t = np.linspace(3*np.pi/2, 2*np.pi, segment_lengths[4, 2]+1, endpoint=True)  # skip last one later
    t = t[0:len(t)-1]
    x0_bound[ind1: ind2] = np.cos(t) * rhole_laa + laa_hole_center_x
    y0_bound[ind1: ind2] = np.sin(t) * rhole_laa + laa_hole_center_y
    return x0_bound, y0_bound

def define_constraints_positions(s1, s2, s3, s4, s5, s6, s7, s8a, s8b, s8c, v1l_x, v1l_y, v1d_x, v1d_y, v1r_x, v1r_y, v1u_x, v1u_y, v2l_x,
                                 v2l_y, v2r_x, v2r_y, v2u_x, v2u_y, v3r_x, v3r_y, v3u_x, v3u_y, v3l_x, v3l_y,
                                 v4r_x, v4r_y, v4u_x, v4u_y, v4d_x, v4d_y, vlaad_x, vlaad_y, vlaau_x, vlaau_y, vlaar_x, vlaar_y, p5_x,
                                 p5_y, p6_x, p6_y, p7_x, p7_y, p8_x, p8_y):
    """Define target (x0,y0) coordinates of regional constraints given segments and template parameters (extreme coordinates of segments)"""
    p_const = s1.shape[0] + s2.shape[0] + s3.shape[0] + s4.shape[0] + s5.shape[0] + s6.shape[0] + s7.shape[0] + s8a.shape[0] + s8b.shape[0] + s8c.shape[0]
    x0_const = np.zeros(p_const)
    y0_const = np.zeros(p_const)
    # s1, vert line, right
    ind1 = 0
    ind2 = s1.shape[0]
    # vert line
    x0_const[ind1:ind2] = v1d_x
    aux = np.linspace(v1d_y, v2u_y, s1.shape[0] + 2, endpoint=True)
    y0_const[ind1:ind2] = aux[1:aux.size - 1]  # skip first and last

    # s2,  bottom line
    ind1 = ind2
    ind2 = ind2 + s2.shape[0]
    # crosswise lines (all with direction starting in the PV ending in the MV). General rule:
    # m = (y2-y1)/(x2-x1)
    # b = y - m*x
    # y = m*x + b   (any x and y in the line)
    aux = np.linspace(v2l_x, v3r_x, s2.size + 2, endpoint=True)
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    m = (v3r_y - v2l_y) / (v3r_x - v2l_x)
    b = v3r_y - m * v3r_x
    aux2 = m * aux + b
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]

    # s3, vert line left
    ind1 = ind2
    ind2 = ind2 + s3.shape[0]
    x0_const[ind1: ind2] = v3u_x
    aux = np.linspace(v3u_y, v4d_y, s3.shape[0] + 2, endpoint=True)
    y0_const[ind1: ind2] = aux[1:aux.size - 1]

    # s4, hori top line
    ind1 = ind2
    ind2 = ind2 + s4.shape[0]
    aux = np.linspace(v4r_x, v1l_x, s4.shape[0] + 2, endpoint=True)
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    m = (v1l_y - v4r_y) / (v1l_x - v4r_x)
    b = v4r_y - m * v4r_x
    aux2 = m * aux + b
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]

    # s5 - line crosswise line from v1r to v5
    ind1 = ind2
    ind2 = ind2 + s5.shape[0]
    m = (p5_y - v1r_y) / (p5_x - v1r_x)
    b = v1r_y - m * v1r_x
    aux = np.linspace(v1r_x, p5_x, s5.shape[0] + 2, endpoint=True)
    aux2 = m * aux + b
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]

    # s6 - line crosswise line from v2r to v6
    ind1 = ind2
    ind2 = ind2 + s6.shape[0]
    m = (p6_y - v2r_y) / (p6_x - v2r_x)
    b = v2r_y - m * v2r_x
    aux = np.linspace(v2r_x, p6_x, s6.shape[0] + 2, endpoint=True)
    aux2 = m * aux + b
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]

    # s7 - line crosswise line from v3l to v7
    ind1 = ind2
    ind2 = ind2 + s7.shape[0]
    m = (p7_y - v3l_y) / (p7_x - v3l_x)
    b = v3l_y - m * v3l_x
    aux = np.linspace(v3l_x, p7_x, s7.shape[0] + 2, endpoint=True)
    aux2 = m * aux + b
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]

    # s8a  - crosswise line from lspv (v4u) to laa
    ind1 = ind2
    ind2 = ind2 + s8a.shape[0]
    aux = np.linspace(v4u_x, vlaad_x, s8a.shape[0] + 2, endpoint=True)
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    m = (vlaad_y - v4u_y) / (vlaad_x - v4u_x)
    b = v4u_y - m * v4u_x
    aux2 = m * aux + b
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]

    # s8b- line crosswise line from vlaau to v8
    ind1 = ind2
    ind2 = ind2 + s8b.shape[0]
    m = (p8_y - vlaau_y) / (p8_x - vlaau_x)
    b = vlaau_y - m * vlaau_x
    if p8_x > vlaau_x:
        print('Warning: v8 is greater (in absolute value) than v_laa_up, consider select a different angle for point V8')
    aux = np.linspace(vlaau_x, p8_x, s8b.shape[0] + 2, endpoint=True)
    aux2 = m * aux + b
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]

    # s8c - line crosswise line from vlaar to v1l
    ind1 = ind2
    ind2 = ind2 + s8c.shape[0]
    m = (v1u_y - vlaar_y) / (v1u_x - vlaar_x)
    b = vlaar_y - m * vlaar_x
    aux = np.linspace(vlaar_x, v1u_x, s8c.shape[0] + 2, endpoint=True)
    aux2 = m * aux + b
    x0_const[ind1: ind2] = aux[1:aux.size - 1]
    y0_const[ind1: ind2] = aux2[1:aux2.size - 1]
    return x0_const, y0_const

def ExtractVTKPoints(mesh):
    """Extract points from vtk structures. Return the Nx3 numpy.array of the vertices."""
    n = mesh.GetNumberOfPoints()
    vertex = np.zeros((n, 3))
    for i in range(n):
        mesh.GetPoint(i, vertex[i, :])
    return vertex


def ExtractVTKTriFaces(mesh):
    """Extract triangular faces from vtkPolyData. Return the Nx3 numpy.array of the faces (make sure there are only triangles)."""
    m = mesh.GetNumberOfCells()
    faces = np.zeros((m, 3), dtype=int)
    for i in range(m):
        ptIDs = vtk.vtkIdList()
        mesh.GetCellPoints(i, ptIDs)
        if ptIDs.GetNumberOfIds() != 3:
            raise Exception("Nontriangular cell!")
        faces[i, 0] = ptIDs.GetId(0)
        faces[i, 1] = ptIDs.GetId(1)
        faces[i, 2] = ptIDs.GetId(2)
    return faces

def ComputeLaplacian(vertex, faces):
    """Calculates the Laplacian of a mesh
    vertex 3xN numpy.array: vertices
    faces 3xM numpy.array: faces"""
    n = vertex.shape[1]
    m = faces.shape[1]

    # compute mesh weight matrix
    W = coo_matrix((n, n))
    for i in np.arange(1, 4, 1):
        i1 = np.mod(i - 1, 3)
        i2 = np.mod(i, 3)
        i3 = np.mod(i + 1, 3)
        pp = vertex[:, faces[i2, :]] - vertex[:, faces[i1, :]]
        qq = vertex[:, faces[i3, :]] - vertex[:, faces[i1, :]]
        # normalize the vectors
        pp = pp / np.sqrt(np.sum(pp ** 2, axis=0))
        qq = qq / np.sqrt(np.sum(qq ** 2, axis=0))
        # compute angles
        ang = np.arccos(np.sum(pp * qq, axis=0))
        W = W + coo_matrix((1 / np.tan(ang), (faces[i2, :], faces[i3, :])), shape=(n, n))
        W = W + coo_matrix((1 / np.tan(ang), (faces[i3, :], faces[i2, :])), shape=(n, n))

    # compute Laplacian
    d = W.sum(axis=0)
    D = dia_matrix((d, 0), shape=(n, n))
    L = D - W
    return L

def flat(m, boundary_ids, x0, y0):
    """Conformal flattening fitting boundary to (x0,y0) coordinate positions"""
    vertex = ExtractVTKPoints(m).T
    faces = ExtractVTKTriFaces(m).T
    n = vertex.shape[1]
    L = ComputeLaplacian(vertex, faces)

    L = L.tolil()
    L[boundary_ids, :] = 0
    for i in range(boundary_ids.shape[0]):
        L[boundary_ids[i], boundary_ids[i]] = 1

    Rx = np.zeros(n)
    Rx[boundary_ids] = x0
    Ry = np.zeros(n)
    Ry[boundary_ids] = y0
    L = L.tocsr()

    result = np.zeros((Rx.size, 2))
    result[:, 0] = linalg_sp.spsolve(L, Rx)  # x
    result[:, 1] = linalg_sp.spsolve(L, Ry)  # y

    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()

    pts.SetNumberOfPoints(n)
    for i in range(n):
        pts.SetPoint(i, result[i, 0], result[i, 1], 0)

    pd.SetPoints(pts)
    pd.SetPolys(m.GetPolys())
    pd.Modified()
    return pd

def flat_w_constraints(m, boundary_ids, constraints_ids, x0_b, y0_b, x0_c, y0_c):
    """ Conformal flattening fitting boundary points to (x0_b,y0_b) coordinate positions
    and additional contraint points to (x0_c,y0_c).
    Solve minimization problem using quadratic programming: https://en.wikipedia.org/wiki/Quadratic_programming"""

    penalization = 1000
    vertex = ExtractVTKPoints(m).T    # 3 x n_vertices
    faces = ExtractVTKTriFaces(m).T
    n = vertex.shape[1]
    L = ComputeLaplacian(vertex, faces)
    L = L.tolil()
    L[boundary_ids, :] = 0     # Not conformal there
    for i in range(boundary_ids.shape[0]):
         L[boundary_ids[i], boundary_ids[i]] = 1

    L = L*penalization

    Rx = np.zeros(n)
    Ry = np.zeros(n)
    Rx[boundary_ids] = x0_b * penalization
    Ry[boundary_ids] = y0_b * penalization

    L = L.tocsr()

    nconstraints = constraints_ids.shape[0]
    M = np.zeros([nconstraints, n])   # M, zero rows except 1 in constraint point
    for i in range(nconstraints):
        M[i, constraints_ids[i]] = 1
    dx = x0_c
    dy = y0_c

    block1 = hstack([L.T.dot(L), M.T])

    zeros_m = coo_matrix(np.zeros([len(dx),len(dx)]))
    block2 = hstack([M, zeros_m])

    C = vstack([block1, block2])
    C = C.tocsr()
    prodx = coo_matrix([L.T.dot(Rx)])
    dxx = coo_matrix([dx])
    cx = hstack([prodx, dxx])

    prody = coo_matrix([L.T.dot(Ry)])
    dyy = coo_matrix([dy])
    cy = hstack([prody, dyy])

    solx = linalg_sp.spsolve(C, cx.T)
    soly = linalg_sp.spsolve(C, cy.T)
    print('There are: ', len(np.argwhere(np.isnan(solx))), ' nans')
    print('There are: ', len(np.argwhere(np.isnan(soly))), ' nans')
    if len(np.argwhere(np.isnan(solx))) > 0:
        print('WARNING!!! matrix is singular. It is probably due to the convergence of 2 different division lines in the same point.')
        print('Trying to assign different 2D possition to same 3D point. Try to create new division lines or increase resolution of mesh.')

    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()

    pts.SetNumberOfPoints(n)
    for i in range(n):
        pts.SetPoint(i, solx[i], soly[i], 0)

    pd.SetPoints(pts)
    pd.SetPolys(m.GetPolys())
    pd.Modified()
    return pd

# From cutter
def find_triangles(p1_id, p2_id, tri):
    tt = (tri - p1_id) * (tri - p2_id)
    return np.where((tt == 0).sum(axis=1) == 2)[0]

def find_celledge_neighbors(tri_id, tri):
    (p1_id, p2_id, p3_id) = tri[tri_id, :]
    t1 = find_triangles(p1_id, p2_id, tri)
    t2 = find_triangles(p1_id, p3_id, tri)
    t3 = find_triangles(p2_id, p3_id, tri)
    t = (set(t1).union(set(t2)).union(set(t3))) - {tri_id}
    return list(t)

def triangle_common_edge(tri1, tri2):
    common_pts = set(tri1).intersection(set(tri2))
    if len(common_pts) < 2:
        return {}
    else:
        return common_pts

def triangles_on_one_line(t1, t2, tri, line):
    edge = triangle_common_edge(tri[t1, :], tri[t2, :])
    on_line = False
    for i in range(line.shape[0] - 1):
        segm = {line[i], line[i + 1]}
        if len(segm - edge) == 0:
            on_line = True
            break
    return on_line

def triangles_on_any_line(t1, t2, tri, lines, m):
    on_line = False
    for line in lines:
        on_line = triangles_on_one_line(t1, t2, tri, line)
        if on_line:
            break
    return on_line

def set_piece_label_from_contours(
    m,
    cont_rspv,
    cont_ripv,
    cont_lspv,
    cont_lipv,
    cont_laa,
    cont_mv, 
    line_textfile,
    file):

    lines = []

    with open(line_textfile, 'r') as f:
        for line in f:
            l = line.replace('\n', '').strip()
            lines.append(np.array([int(x) for x in l.split(' ')]))

    # extract connectivity
    tri = np.zeros([m.GetNumberOfCells(), 3], dtype=np.int64)

    for i in range(tri.shape[0]):
        ids = m.GetCell(i).GetPointIds()
        for j in range(3):
            tri[i, j] = ids.GetId(j)
    trilabel = np.zeros(m.GetNumberOfCells(), dtype=np.int64)
    region_id = 0
    for i in range(m.GetNumberOfCells()):
        if trilabel[i] == 0:
            tri_stack = [i]  # triangles to process
            region_id = region_id + 1

            while tri_stack:  # while not empty
                tri_id = tri_stack.pop()
                if (trilabel[tri_id] == 0):  # if not labeled yet
                    trilabel[tri_id] = region_id
                    neighb = find_celledge_neighbors(tri_id, tri)

                    for j in range(len(neighb)):
                        if trilabel[neighb[j]] == 0:
                            # see if the triangles tri_id and neighb[j] are on the different sides of the line
                            # i.e. if they share any pair of points of the line
                            if not triangles_on_any_line(tri_id, neighb[j], tri, lines, m):
                                tri_stack.append(neighb[j])

    trilabel_vtkarray = numpy_to_vtk(trilabel)
    trilabel_vtkarray.SetName('region')
    m.GetCellData().AddArray(trilabel_vtkarray)
    def contour_point_ids(cont):
        return set(cont)

    contours = {
        "RSPV": contour_point_ids(cont_rspv),
        "RIPV": contour_point_ids(cont_ripv),
        "LSPV": contour_point_ids(cont_lspv),
        "LIPV": contour_point_ids(cont_lipv),
        "LAA": contour_point_ids(cont_laa),
        "MV": contour_point_ids(cont_mv),
    }

    # --- Extract mesh connectivity ---
    n_cells = m.GetNumberOfCells()


    region_ids = np.unique(trilabel)
    region_ids = region_ids[region_ids != 0]

    standard_regions = np.zeros(n_cells, dtype=np.int64)

    # --- Process each region ---
    for r in region_ids:

        region_cells = np.where(trilabel == r)[0]
        region_points = set(tri[region_cells].flatten())

        touched = []

        for name, contour_pts in contours.items():
            if len(region_points.intersection(contour_pts)) > 0:
                touched.append(name)

        touched = set(touched)
        print(f"Region {r} touches: {touched}")

        # ---- Classification Rules ----
        if {"RSPV", "RIPV", "LSPV", "LIPV"}.issubset(touched):
            label = 5  # posterior
        elif {"MV", "RIPV", "RSPV"}.issubset(touched):
            label = 1  # septal
        elif {"RSPV", "LSPV", "LAA"}.issubset(touched):
            label = 6 # roof
        elif {"RIPV", "LIPV", "MV"}.issubset(touched):
            label = 2 # inferior
        elif {"LIPV", "LSPV", "LAA" , "MV"}.issubset(touched):
            label = 3 # lateral
        elif {"RSPV", "LAA" , "MV"}.issubset(touched):
            label = 4 # anterior
        else:
            label = 0  # unknown

        standard_regions[region_cells] = label

    # --- Save region array ---
    cellarray = numpy_to_vtk(standard_regions)
    cellarray.SetName("region")
    m.GetCellData().AddArray(cellarray)

    return m
