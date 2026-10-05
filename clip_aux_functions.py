# # Code adapted from https://github.com/catactg/SUM
# # Related publication describing semi-automatic PV clipping algorithm: "Benchmark for Algorithms Segmenting the Left Atrium From 3D CT and MRI Datasets",
# # Catalina Tobon-Gomez et al., IEEE transactions on medical imaging, 2015.

from aux_functions import *
from vmtkfunctions import *
import seedselector
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import os

def seed_interactor(surface, points):
    """Interactor for seed selection. Needs VMTK"""
    computer = seedselector.vmtkPickPointSeedSelector()
    computer.SetSurface(surface)
    computer.SetVisualizationPoints(points)
    computer.Execute()
    return computer.GetSourceSeedIds()

def select_seeds(surface, labels, surfacefileout, vis=0, laa=1):
    """Select 4 seeds (1 per vein) and a 5th one if laa = 1"""
    if laa ==1:
        labelsrange = [76.0, 77.0, 79.0, 78.0, 37.0]
        nseeds = 5
    
    else:
        # for each PV
        labelsrange = [76.0, 77.0, 79.0, 78.0]
        nseeds = 4

    seeds = seed_interactor(surface, vtk.vtkPoints())
    # create the pointset
    newpoints = vtk.vtkPoints()
    newvertices = vtk.vtkCellArray()

    # create array on seeds with ground truth (GT) labels
    gtlabels_array = vtk.vtkDoubleArray()
    gtlabels_array.SetName(labels)

    if not seeds.GetNumberOfIds() == nseeds:
        print('You should select exactly', nseeds, ' seeds. Try again!')
        seeds = seed_interactor(surface, vtk.vtkPoints())

    for s in range(seeds.GetNumberOfIds()):
        branchlabel = labelsrange[s]
        point = surface.GetPoint(seeds.GetId(s))
        pid = newpoints.InsertNextPoint(point)
        gtlabels_array.InsertNextValue(branchlabel)
        # Create the topology of the point (a vertex)
        newvertices.InsertNextCell(1)
        newvertices.InsertCellPoint(pid)
    pointspd = vtk.vtkPolyData()
    pointspd.SetPoints(newpoints)
    pointspd.SetVerts(newvertices)
    pointspd.GetPointData().AddArray(gtlabels_array)

    if vis==1:
        pointsgplyh = generateglyph(pointspd)
        visualise_default(pointsgplyh, surface, 'seeds', labels, 36, 79)
    writevtp(pointspd, surfacefileout)

def change_seeds(surface, labels, seedsfile, seed_order, vis=1,laa=1):
    """Change the seed numbers in the seeds file to newseeds"""

    seeds_dict = {"RSPV": 76,"RIPV": 77, "LIPV": 79, "LSPV": 78,  "LAA": 37}
    pointspd = readvtp(seedsfile)
    points = pointspd.GetPoints()
    newseeds = seed_interactor(surface, points)
    vertices = pointspd.GetVerts()
    gtlabels_array = pointspd.GetPointData().GetArray(labels)  

    # Search for the point with the target value
    for j, target_value in enumerate(seed_order):
        for i in range(pointspd.GetNumberOfPoints()):
            if gtlabels_array.GetValue(i) == seeds_dict[target_value]:
                coord = list(points.GetPoint(i))
                new_coord = surface.GetPoint(newseeds.GetId(j)) 
                points.SetPoint(i, new_coord)
                points.Modified() 
                break

    pointspd_new = vtk.vtkPolyData()
    pointspd_new.SetPoints(points)
    pointspd_new.SetVerts(vertices)
    pointspd_new.GetPointData().AddArray(gtlabels_array)
    
    if vis==1:
        pointsgplyh = generateglyph(pointspd_new)
        visualise_default(pointsgplyh, surface, 'seeds', labels, 36, 79)
    writevtp(pointspd_new, seedsfile)


def getregionslabels():
    """Return dictionary linking regionids to anatomical locations."""
    regionslabels = {'body': 36,
                     'laa': 37,
                     'pv2': 76,
                     'pv1': 77,
                     'pv3': 78,
                     'pv4': 79}
    return regionslabels

def create_autolabels(surface, ref,  arrayname, value):
    """Create autolabels scalar array (mark PVs using branch labels) and add it to surface """
    #visualise_two_meshes(surface, ref)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()

    array = surface.GetPointData().GetArray(arrayname)
    
    for i in range(ref.GetNumberOfPoints()):
        point = ref.GetPoint(i)
        closestpoint_id = locator.FindClosestPoint(point)
        array.SetValue(closestpoint_id, value)

        ids = vtk.vtkIdList()
        locator.FindPointsWithinRadius(1, point, ids)
        for j in range(ids.GetNumberOfIds()):
            pid = ids.GetId(j)
            array.SetValue(pid, value)
    return surface

def centroidofcentroids(edges):
    # compute centroids of each edge
    # find average point
    acumvector = [0,0,0]
    rn = countregions(edges)
    print('number of edges',rn)
    for r in range(rn):
        oneedge = extractconnectedregion(edges,r)
        onecentroid = pointset_centreofmass(oneedge)
        acumvector = acumvectors(acumvector,onecentroid)
    finalcentroid = dividevector(acumvector,rn)
    return finalcentroid

def pv_LAA_centerlines(inputfile, seedsfile, outfile, pvends=1):
    """ Create 5 pairs of centerlines, each one starting from each PV (or LAA) seed and going to the 2 opposite
    (other side) PVs"""

    surface = readvtk(inputfile)
    points = np.loadtxt(seedsfile, delimiter=',').tolist()

    print('Processing RSPV seed:')
    cl1 = vmtkcenterlines(surface, points[0], points[2] + points[3], pvends)
    print('\n \nProcessing RIPV seed:')
    cl2 = vmtkcenterlines(surface, points[1], points[2] + points[3], pvends)
    print('\n \nProcessing LIPV seed:')
    cl3 = vmtkcenterlines(surface, points[2], points[0] + points[1], pvends)
    print('\n \nProcessing LSPV seed:')
    cl4 = vmtkcenterlines(surface, points[3], points[0] + points[1], pvends)
    print('\n \nProcessing LAA seed:')
    cl5 = vmtkcenterlines(surface, points[4], points[0] + points[1], pvends)

    writevtp(cl1, outfile + 'clraw21.vtp')
    writevtp(cl2, outfile + 'clraw22.vtp')
    writevtp(cl3, outfile + 'clraw23.vtp')
    writevtp(cl4, outfile + 'clraw24.vtp')
    writevtp(cl5, outfile + 'clraw25.vtp')

def intersectwithline(surface, p1, p2):
    """Given surface and line defined by 2 points (p1,p2), return insersecting points"""
    tree = vtk.vtkOBBTree()
    tree.SetDataSet(surface)
    tree.BuildLocator()

    intersectPoints = vtk.vtkPoints()
    intersectCells = vtk.vtkIdList()

    tolerance=1.e-3
    tree.SetTolerance(tolerance)
    tree.IntersectWithLine(p1, p2, intersectPoints, intersectCells)
    return intersectPoints

def furthest_point_to_polydata(pointset,refpoint):
    """Given set of points and ref point, select furthest point using euclidean dist"""
    refdist = 0
    for i in range(pointset.GetNumberOfPoints()):
        dist = euclideandistance(pointset.GetPoint(i),refpoint)
        if dist > refdist:
            refdist = dist
            selectedpointid = i
    return pointset.GetPoint(selectedpointid)

def computelengthalongvector(polydata, refpoint, vector):
    # polydata should be a closed surface

    # intersect with line
    point1 = refpoint
    point2 = sumvectors(refpoint,1000,vector) # far away point
    intersectpoints = intersectwithline(polydata,point1,point2)
    furthestpoint1 = furthest_point_to_polydata(intersectpoints,refpoint)

    # intersect with line the other way
    point1 = refpoint
    point2 = sumvectors(refpoint,-1000,vector) # far away point
    intersectpoints = intersectwithline(polydata,point1,point2)
    furthestpoint2 = furthest_point_to_polydata(intersectpoints,furthestpoint1)
    length = euclideandistance(furthestpoint1,furthestpoint2)
    return length, furthestpoint1, furthestpoint2
    
def clip_vein(surface,cl,clippointid):
    """Clip the vein at clippoint"""
    clippoint0 = cl.GetPoint(clippointid)
    clipnormal = (np.array(cl.GetPoint(clippointid+1)) - np.array(cl.GetPoint(clippointid)))
    possvein = planeclip(surface, clippoint0, clipnormal)
    vein = extractclosestpointregion(possvein,clippoint0)
    return vein

def skippoints(polydata, nskippoints):
    """Generate a single cell line from points in idlist."""
    # derive number of nodes
    numberofnodes = polydata.GetNumberOfPoints() - nskippoints

    # define points and line
    points = vtk.vtkPoints()
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(numberofnodes)

    # assign id and x,y,z coordinates
    for i in range(nskippoints,polydata.GetNumberOfPoints()):
        pointid = i - nskippoints
        polyline.GetPointIds().SetId(pointid,pointid)
        point = polydata.GetPoint(i)
        points.InsertNextPoint(point)

    # define cell
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    # add to polydata
    polyout = vtk.vtkPolyData()
    polyout.SetPoints(points)
    polyout.SetLines(cells)
    if not vtk.vtkVersion.GetVTKMajorVersion() > 5:
        polyout.Update()
    return polyout

def centerline_length(cl):
    pts = vtk.util.numpy_support.vtk_to_numpy(cl.GetPoints().GetData())
    if pts.shape[0] < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return np.sqrt((diffs**2).sum(axis=1)).sum()

def compute_path_length(path):
    if path.GetNumberOfPoints() < 2:
        return 0.0
    pts = vtk.util.numpy_support.vtk_to_numpy(path.GetPoints().GetData())
    return np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()

def veins_overlap(mesh, v1, v2, tol=4):
    loc_mesh = vtk.vtkPointLocator()
    loc_mesh.SetDataSet(mesh)
    loc_mesh.BuildLocator()
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(v2)
    loc.BuildLocator()
    min_dist = 1000
    for i in range(v1.GetNumberOfPoints()):
        p = v1.GetPoint(i)
        pid = loc.FindClosestPoint(p)
        if pid >= 0:
            q = v2.GetPoint(pid)
            if vtk.vtkMath.Distance2BetweenPoints(p, q) < min_dist:
                min_dist = vtk.vtkMath.Distance2BetweenPoints(p, q)
                if min_dist < tol**2:
                    shortest_path = find_create_path(mesh, loc_mesh.FindClosestPoint(p), loc_mesh.FindClosestPoint(q))
                    dist_on_mesh = compute_path_length(shortest_path)
                    if dist_on_mesh < tol:
                        print(f"[INFO] Veins overlap detected: distance {dist_on_mesh} < tolerance {tol}")
                        return True
    return False

def decide_which_to_move(bi, bj):   
    seeds_dict = {76: "RSPV", 77:"RIPV", 79:"LIPV", 78:"LSPV", 37:"LAA"}
    branchlabel = [0, 77, 76, 78, 79, 37]  # RSPV, RIPV, LSPV, LIPV, LAA
    print(f"[DEBUG] Deciding which vein to move: {seeds_dict[branchlabel[bi['k']]]} area {bi['area'].GetValue(bi['clippointid'])}, {seeds_dict[branchlabel[bj['k']]]} area {bj['area'].GetValue(bj['clippointid'])}")
    if bi['area'].GetValue(bi['clippointid']) > 800 and bi['area'].GetValue(bi['clippointid']) > bj['area'].GetValue(bj['clippointid']):
        keep, move = bj, bi
    elif bj['area'].GetValue(bj['clippointid']) > 800 or  bi['area'].GetValue(bi['clippointid']) < 100:
        keep, move = bi, bj
    else:
        keep, move = bj, bi
    return keep, move

def check_cut_diameter(signal, index, threshold, threshold_factor=1):
    print(f"[DEBUG] Checking diameter at index {index}: {signal[index]} < {threshold * threshold_factor} -> {signal[index] < (threshold * threshold_factor)}")
    return signal[index] < threshold * threshold_factor

def check_peak_validity(signal, peak, threshold, n):
    """
    Checks whether a peak in a signal is valid based on a threshold.
    If the peak is preceeded by a huge pit, it means that the increase is followed by a decrease and so its just a local peak
    """

    for i in range(peak-1, min(len(signal),peak+n+1)):
        if (signal[i] < (-abs(threshold))) :
            print(f"[DEBUG] Peak at index {peak} is not valid, value {signal[i]} is below threshold {-abs(threshold)}")
            return False
            
    return True

def intersect_mesh_with_plane(polydata, origin, normal):
    """
    Intersect vtkPolyData with a plane.
    """

    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(polydata)
    cutter.Update()

    return cutter.GetOutput()


def ordered_contour_points(polydata):
    """
    Extract ordered contour points from vtkCutter output.
    """

    pts = vtk_to_numpy(polydata.GetPoints().GetData())
    center = pts.mean(axis=0)

    # project to YZ plane
    yz = pts[:, 1:] - center[1:] 
    
    angles = np.arctan2( yz[:, 1], yz[:, 0] )
    order = np.argsort(angles)

    return pts[order]


def maximum_diameter_mesh_fixed_plane(polydata, n_angles=100):
    """
    Compute minimum diameter through COM
    on fixed plane x = x_COM.
    """

    com_filter = vtk.vtkCenterOfMass()
    com_filter.SetInputData(polydata)
    com_filter.Update()

    center = np.array(com_filter.GetCenter())

    # intersect mesh with plane x = x_com
    contour = intersect_mesh_with_plane(
        polydata,
        origin=center,
        normal=[1, 0, 0]
    )

    pts = ordered_contour_points(contour)

    # close contour
    pts = np.vstack([pts, pts[0]])

    center2d = center[1:]

    max_dist = -np.inf
    best_pair = None

    # fixed deterministic directions
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)

    for theta in angles:

        d = np.array([
            np.cos(theta),
            np.sin(theta)
        ])

        intersections = []

        # intersect line with polygon edges
        for i in range(len(pts) - 1):

            a = pts[i][1:] - center2d
            b = pts[i + 1][1:] - center2d

            edge = b - a

            A = np.column_stack([d, -edge])

            det = np.linalg.det(A)

            if abs(det) < 1e-12:
                continue

            try:
                s, u = np.linalg.solve(A, a)
            except np.linalg.LinAlgError:
                continue

            if 0 <= u <= 1:
                intersections.append((s, i, u))

        if len(intersections) < 2:
            continue

        intersections.sort(key=lambda y: y[0])

        s1, i1, u1 = intersections[0]
        s2, i2, u2 = intersections[-1]

        p1 = pts[i1] + u1 * (pts[i1 + 1] - pts[i1])
        p2 = pts[i2] + u2 * (pts[i2 + 1] - pts[i2])

        dist = np.linalg.norm(p1 - p2)

        if dist > max_dist:
            max_dist = dist
            best_pair = (p1, p2)

    return max_dist, best_pair, contour

def compute_diameters_3d(points):
    """
    points: (N,3) planar section

    returns:
        max_diameter, min_diameter
        (p1, p2), (q1, q2)
        filtered points
    """

    # --- MAX DIAMETER ---
    dists = squareform(pdist(points))
    i, j = np.unravel_index(np.argmax(dists), dists.shape)

    p1, p2 = points[i], points[j]
    L = np.linalg.norm(p2 - p1)
    d = (p2 - p1) / L

    # --- ESTIMATE PLANE NORMAL ---
    centered = points - points.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered)
    n_plane = Vt[2]  # normal

    # --- FILTER CENTRAL BAND ---
    t = np.dot(points - p1, d)
    mask = (t >= 0.4 * L) & (t <= 0.6 * L)
    filtered = points[mask]

    if len(filtered) < 2:
        return 10000, -1, (p1, p2), (None, None), filtered

    # --- SPLITTING DIRECTION (IN-PLANE, ⟂ to d) ---
    n_split = np.cross(d, n_plane)
    n_split /= np.linalg.norm(n_split)

    m = 0.5 * (p1 + p2)

    side_A = filtered[np.dot(filtered - m, n_split) >= 0]
    side_B = filtered[np.dot(filtered - m, n_split) < 0]

    if len(side_A) == 0 or len(side_B) == 0:
        return 10000, -1, (p1, p2), (None, None), filtered

    # --- MIN DISTANCE BETWEEN SIDES ---
    D = cdist(side_A, side_B)
    idx = np.unravel_index(np.argmin(D), D.shape)

    q1 = side_A[idx[0]]
    q2 = side_B[idx[1]]
    max_diameter = L
    min_diameter = D[idx]

    return max_diameter, min_diameter, (p1, p2), (q1, q2), filtered

def find_clippoint_id(signal, filtered_signal, diameter_signal, max_diam_mesh, veinNumber, threshold_percentile=90):
    """
    Find the indices of the first value in the filtered signal that is above a dynamic threshold.

    Parameters:
    filtered_signal (np.array): The filtered signal array.
    threshold_factor (float): The percentile to which to set the threshold.
    veinnumber is used to adapt the clipping when it's the laa, because this one works differently

    Returns:
    indices (list): The indices of all suitable peaks.

    """
    # Calculate the mean and standard deviation of the filtered signal
    mean = np.mean(filtered_signal)
    std_dev = np.std(filtered_signal)

    # Set the dynamic threshold
    threshold = mean + 0.5 * std_dev
    diam_factor = 0.7

    # Iterate through the filtered signal to find the peaks of the signal 
    peaks, _ = find_peaks(filtered_signal)
    print(f"[DEBUG] Found peaks at indices: {peaks}")

    # Find the first local maximum
    peakfound = False 
    loopcount = 0 
    indices = []
    while peakfound == False : #! CHECK FOR WHILE TRUE AND STOP IF PEAKS IS EMPTY 
        for peak in peaks : 
            print(f"[DEBUG] Evaluating peak at index {peak} with value {filtered_signal[peak]} against threshold {threshold} and maximal cut diameter {max_diam_mesh * diam_factor}")
            if filtered_signal[peak] > threshold  and check_cut_diameter(diameter_signal, peak, max_diam_mesh, diam_factor):                
                print(f"[DEBUG] Peak found at index {peak} with value {filtered_signal[peak]}")
                if check_peak_validity(filtered_signal, peak, threshold, 5):  
                    indices.append(peak)
                    peakfound = True
            
        loopcount += 1 
        if loopcount <= 10: 
            threshold -= 0.1 * std_dev
        elif loopcount <= 16 :
            diam_factor += 0.1        # Decrease threshold to find the first local maximum above threshold
        else :
            first_local_maximum_value = filtered_signal[peaks[1]]
            print(f"[INFO] The second local maximum is at index {peaks[1]} with a value of {first_local_maximum_value}.")
            return [peaks[1]]  # Return the second local maximum if no peak is found after 16 iterations
            
    return indices

def apply_high_pass_filter(signal):
    # Normalize the signal
    normalized_signal = (signal - np.mean(signal)) / np.std(signal)
    
    # Define a high-pass filter with length 5
    high_pass_filter = np.array([4, 2, 0,-2, -4])  
    filtered_signal = np.convolve(signal, high_pass_filter, mode='valid')

    #crop the signal before analysis 
    filtered_signal = filtered_signal[:130]
    normalized_signal = normalized_signal[:130]
   
    #!return filtered_signal, normalized_signal
    return filtered_signal, signal

def contour_area(contour):
    """
    Calculate the area enclosed by an ordered, planar Nx3 contour.
    """
    if len(contour) < 3:
        return 0.0

    # Remove duplicate closing point if present
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]

    # Center the points
    centered = contour - contour.mean(axis=0)

    # Find the two directions spanning the contour plane
    _, _, vh = np.linalg.svd(centered)

    u = vh[0]
    v = vh[1]

    # Project 3D points onto the plane
    x = centered @ u
    y = centered @ v

    # Shoelace formula
    return 0.5 * abs(
        np.dot(x, np.roll(y, -1))
        - np.dot(y, np.roll(x, -1))
    )

def polygon_to_polydata(points_np):
    """
    Wrap a closed polygon in a vtkPolyData for triangulation.
    
    points_np : Nx3 numpy array
        Ordered points of the polygon.
    
    Returns
    -------
    vtkPolyData containing a single polygon
    """
    # Create vtkPoints
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(points_np))
    
    # Create vtkPolygon
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(points_np.shape[0])
    for i in range(points_np.shape[0]):
        polygon.GetPointIds().SetId(i, i)
    
    # Create cell array and insert polygon
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polygon)
    
    # Create vtkPolyData
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_pts)
    polydata.SetPolys(cells)
    
    return polydata

def get_centerline_tangent(cl, point_id):

    n = cl.GetNumberOfPoints()

    if point_id == 0:
        p0 = np.array(cl.GetPoint(0))
        p1 = np.array(cl.GetPoint(1))

    elif point_id == n - 1:
        p0 = np.array(cl.GetPoint(n - 2))
        p1 = np.array(cl.GetPoint(n - 1))

    else:
        p0 = np.array(cl.GetPoint(point_id - 1))
        p1 = np.array(cl.GetPoint(point_id + 1))

    tangent = p1 - p0
    tangent /= np.linalg.norm(tangent)

    return tangent

def point_in_contour(point, contour, normal):

    point = np.asarray(point, dtype=float)
    contour = np.asarray(contour, dtype=float)
    normal = np.asarray(normal, dtype=float)

    normal /= np.linalg.norm(normal)

    # Construct basis of cutting plane
    ref = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)

    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    # Project onto plane
    relative = contour - point

    x = relative @ u
    y = relative @ v

    # Point is (0,0)
    inside = False

    j = len(x) - 1

    for i in range(len(x)):

        # Does the edge cross y=0?
        if (y[i] > 0) != (y[j] > 0):

            # x coordinate where edge intersects y=0
            x_cross = (
                x[i]
                + (x[j] - x[i])
                * (-y[i])
                / (y[j] - y[i])
            )

            if x_cross > 0:
                inside = not inside

        j = i

    return inside
def polyline_to_polydata(polydata, cell_id):

    cell = polydata.GetCell(cell_id)
    ids = cell.GetPointIds()

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    id_map = {}

    for j in range(ids.GetNumberOfIds()):
        old_id = ids.GetId(j)
        new_id = points.InsertNextPoint(polydata.GetPoint(old_id))
        id_map[old_id] = new_id

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(ids.GetNumberOfIds())

    for j in range(ids.GetNumberOfIds()):
        old_id = ids.GetId(j)
        polyline.GetPointIds().SetId(j, id_map[old_id])

    lines.InsertNextCell(polyline)

    output = vtk.vtkPolyData()
    output.SetPoints(points)
    output.SetLines(lines)

    return output
def get_outer_contour(polydata, clippoint, normal):

    stripper = vtk.vtkStripper()
    stripper.SetInputData(polydata)
    stripper.Update()

    stripped = stripper.GetOutput()

    section = None
    contours = []
    if stripped.GetNumberOfCells() > 1:
        min_area = 1000000 
        for i in range(stripped.GetNumberOfCells()):
            contour = stripped.GetCell(i)
            contour_pts = vtk_to_numpy(contour.GetPoints().GetData())
            if point_in_contour(clippoint, contour_pts, normal):
                area = contour_area(contour_pts)
                if area < min_area:
                    min_area = area
                    section = polyline_to_polydata(stripped, i)
    else:
        section = polyline_to_polydata(stripped,0)
    return section

def clip_veins_sections_and_LAA(inputfile, sufixfile, clspacing, maxslope, skippointsfactor, highslope, bumpcriterion, eams, seedsfile):

    surface = vmtksurfacereader(inputfile)
    seeds_dict = {76: "RSPV", 77:"RIPV", 79:"LIPV", 78:"LSPV", 37:"LAA"}
    branchlabel = [0, 77, 76, 78, 79, 37]  # RSPV, RIPV, LSPV, LIPV, LAA
    branch_array = vtk.vtkDoubleArray()
    branch_array.SetName('autolabels')
    branch_array.SetNumberOfTuples(surface.GetNumberOfPoints())
    surface.GetPointData().AddArray(branch_array)

    for i in range(surface.GetNumberOfPoints()):
        branch_array.SetValue(i, round(36))  # init mitral valve
    max_diam_mesh, points, contour = maximum_diameter_mesh_fixed_plane(surface)
    #visualize_minimum_diameter_vtk(surface, points, contour)
    branch_infos = []
    clippoint_dict = {}
    # ---- main loop (unchanged except storing results) ----
    for k in range(1, 6):
        print("branchlabel", branchlabel[k], seeds_dict[branchlabel[k]])
        cl = readvtp(sufixfile + 'clraw2' + str(k) + '.vtp')
        cl = vmtkcenterlineresampling(cl, clspacing)
        cl = vmtkcenterlinesmoothing(cl)
        cl = vmtkbranchextractor(cl)
        writevtp(cl, sufixfile + 'clbranch' + str(k) + '.vtp')

        cl = cellthreshold(cl, 'GroupIds', 0, 0)
        cl = vmtkcenterlinemerge(cl)
        cl = vmtkcenterlineresampling(cl, clspacing)
        cl = vmtkcenterlineattributes(cl)
        
        nskippoints = round(skippointsfactor * cl.GetNumberOfPoints())
        cl2 = skippoints(cl, int(nskippoints))
        cl2 = vmtkcenterlineresampling(cl2, clspacing)
        cl2 = vmtkcenterlineattributes(cl2)
        writevtp(cl2, sufixfile +'clvein' + str(k) + '.vtp')
        
        if k == 5:
            all_sections = vtk.vtkAppendPolyData()
            areaarray = vtk.vtkDoubleArray()
            area_values = []
            for i in range(cl2.GetNumberOfPoints()):
                origin = np.array(cl2.GetPoint(i))
                normal = get_centerline_tangent(cl2, i)

                plane = vtk.vtkPlane()
                plane.SetOrigin(origin)
                plane.SetNormal(normal)

                cutter = vtk.vtkCutter()
                cutter.SetInputData(surface)
                cutter.SetCutFunction(plane)
                cutter.Update()
                sectionnew = cutter.GetOutput()
                sectionnew = get_outer_contour(sectionnew, np.array(cl2.GetPoint(i)), normal)
                all_sections.AddInputData(sectionnew)

                area = contour_area(vtk_to_numpy(sectionnew.GetPoints().GetData()))
                areaarray.InsertNextValue(area)
                area_values.append(area)
            areaarray.SetName('CenterlineSectionArea')
            all_sections.Update()
            sections = vtk.vtkPolyData()
            sections.DeepCopy(all_sections.GetOutput())
            writevtp(sections, sufixfile + 'clsections' + str(k) + '.vtp')
        else:
            sections = vmtkcenterlinesections(surface, cl2)
            writevtp(sections, sufixfile + 'clsections' + str(k) + '.vtp')
            closed = sections.GetCellData().GetArray('CenterlineSectionClosed')
            maxsize = sections.GetCellData().GetArray('CenterlineSectionMaxSize')
            areaarray = sections.GetCellData().GetArray('CenterlineSectionArea')
            area_values = [areaarray.GetValue(i) for i in range(1, sections.GetNumberOfCells()) if closed.GetValue(i - 1)]
        min_diameter_values = []
        max_diameter_values = []
        if k == 5 or eams:
            print(f"Using new clipping method for {seeds_dict[branchlabel[k]]}")
            for i in range(1, sections.GetNumberOfCells()):
                if k==5 or closed.GetValue(i - 1):
                    max_diameter, min_diameter, _, _, _ = compute_diameters_3d(vtk_to_numpy(sections.GetCell(i).GetPoints().GetData()))
                    min_diameter_values.append(min_diameter)
                    max_diameter_values.append(max_diameter)
            
            #filtering the cross section along centerline signal to extract peaks 
            filtered_signal, normalized_signal = apply_high_pass_filter(area_values)
            filtered_signal = savgol_filter(filtered_signal, window_length=11, polyorder=3)

            # Find the clipping point based on the filtered signal
            clippointids = find_clippoint_id(area_values, filtered_signal, max_diameter_values, max_diam_mesh, k)
            clippointid = max(clippointids[-1], 4)

            print(f"[INFO] {seeds_dict[branchlabel[k]]} clipping point ID determined at: {clippointid}")
        else:
            highcount = 0
            nbump = round(bumpcriterion * cl2.GetNumberOfPoints())
            for i in range(1, sections.GetNumberOfCells()):
                if closed.GetValue(i-1):
                    slope = (maxsize.GetValue(i) - maxsize.GetValue(i-1)) / clspacing
                    if slope > highslope: highcount += 1
                    else: highcount = 0
                    if slope > maxslope: break
                    elif slope > highslope and highcount == nbump: break

            clippointid = i - (highcount if highcount > 0 else 1)
            clippointid += int(nskippoints)

        vein = clip_vein(surface, cl2, clippointid)
        length = centerline_length(vein)

        branch_infos.append({
            'k': k, 'cl': cl2, 'clippointid': clippointid,
            'vein': vein, 'length': length, 'sections': sections, 'area': areaarray, 'clippointoptions': clippointids[:-1]
        })
        np.savetxt(sufixfile + f'clippointid{k}.csv', np.array([clippointid]), fmt='%i')

    for i in range(len(branch_infos)):
        for j in range(i+1, len(branch_infos)):
            keep, move = branch_infos[i], branch_infos[j]
            count = 0
            while veins_overlap(surface, keep['vein'], move['vein'], tol=4): 
                if count == 0:
                    print(f"Overlapping veins {seeds_dict[branchlabel[keep['k']]]} and {seeds_dict[branchlabel[move['k']]]}")                
                keep, move = decide_which_to_move(keep, move)
                print(f"[INFO] Moving branch {seeds_dict[branchlabel[move['k']]]} from clippointid {move['clippointid']} to {move['clippointid'] - 1}")
                if move['clippointid'] == 3:
                    if keep['clippointid'] == 3:
                        print(f"[WARNING] Could not resolve overlap for branches {keep['k']} and {move['k']}, both reached end of centerline.")
                        break
                    else:
                        keep, move = move, keep
                move['clippointid'] -= 1
                move['vein'] = clip_vein(surface, move['cl'], move['clippointid'])
                
                np.savetxt(sufixfile + f'clippointid{move["k"]}.csv', np.array([move['clippointid']]), fmt='%i')
                if move['vein'].GetNumberOfPoints() == 0:
                    print(f"[WARNING] Empty vein for branch {move['k']} at clippointid={move['clippointid']}")
                    break
                if count > 2 and ((len(keep['clippointoptions']) > 0 and keep['clippointoptions'][-1]>4) or (len(move['clippointoptions']) > 0) and move['clippointoptions'][-1]>4):
                    keep, move = branch_infos[i], branch_infos[j]
                    if len(keep['clippointoptions']) > 0 and keep['clippointoptions'][-1]>4: 
                        clippointid_keep_new = max(keep['clippointoptions'][-1], 4)
                        print("Veins overlapping, reselecting clippoint of vein ",seeds_dict[branchlabel[keep['k']]], ' choosing id ',clippointid_keep_new, ' instead of ', keep['clippointid']) 
                        vein_keep_new = clip_vein(surface, keep['cl'], clippointid_keep_new)
                        if len(move['clippointoptions']) > 0 and move['clippointoptions'][-1]>4:
                            clippointid_move_new = max(move['clippointoptions'][-1], 4)
                            print("Veins overlapping, reselecting clippoint of vein ", seeds_dict[branchlabel[move['k']]], ' choosing id ',clippointid_move_new, ' instead of ', move['clippointid']) 
                            vein_move_new = clip_vein(surface, move['cl'], clippointid_move_new)
                            if veins_overlap(surface, keep['vein'], vein_move_new):
                                if veins_overlap(surface, vein_keep_new, move['vein']):
                                    keep['vein'] = vein_keep_new
                                    keep['clippointid'] = clippointid_keep_new
                                    keep['clippointoptions'] = keep['clippointoptions'][:-1]
                                    move['vein'] = vein_move_new
                                    move['clippointid'] = clippointid_move_new
                                    move['clippointoptions'] = move['clippointoptions'][:-1]
                                else:
                                    keep['vein'] = vein_keep_new
                                    keep['clippointid'] = clippointid_keep_new
                                    keep['clippointoptions'] = keep['clippointoptions'][:-1]
                            else:
                                move['vein'] = vein_move_new
                                move['clippointid'] = clippointid_move_new
                                move['clippointoptions'] = move['clippointoptions'][:-1]
                        else:
                            keep['vein'] = vein_keep_new
                            keep['clippointid'] = clippointid_keep_new
                            keep['clippointoptions'] = keep['clippointoptions'][:-1]
                    elif len(move['clippointoptions']) > 0 and move['clippointoptions'][-1]>4:
                        clippointid_move_new = max(move['clippointoptions'][-1], 4)
                        print("Veins overlapping, reselecting clippoint of vein ", seeds_dict[branchlabel[move['k']]], ' choosing id ',clippointid_move_new, ' instead of ', move['clippointid']) 
                        vein_move_new = clip_vein(surface, move['cl'], clippointid_move_new)
                        move['vein'] = vein_move_new
                        move['clippointid'] = clippointid_move_new
                        move['clippointoptions'] = move['clippointoptions'][:-1]
                    count = 0
                else:
                    count += 1 
                    #visualise_veins(keep['vein'], move['vein'], surface)
            
            keep['vein'] = clip_vein(surface, keep['cl'], keep['clippointid'])
            keep['length'] = centerline_length(keep['vein'])
            np.savetxt(sufixfile + f'clippointid{keep["k"]}.csv', np.array([keep['clippointid']]), fmt='%i')
        
            move['vein'] = clip_vein(surface, move['cl'], move['clippointid'])
            move['length'] = centerline_length(move['vein'])
            np.savetxt(sufixfile + f'clippointid{move["k"]}.csv', np.array([move['clippointid']]), fmt='%i')
            #visualise_veins(keep['vein'], move['vein'], surface)

    # ---- apply autolabels ----
    for info in branch_infos:
        print(f"Branch {info['k']} final clippointid: {info['clippointid']}")       
        surface = create_autolabels(surface, info['vein'], 'autolabels', round(branchlabel[info['k']]))
    writevtp(surface, sufixfile + 'autolabels.vtp')

def clip_vein_endpoint_and_LAA_save_planes(surface, ifile_sufix, targetdistance, specialvein=0, specialdist=0):
    """Clip vein the targetdistance away from the body. Clip also the LAA at specialdist.
    Return the clip planes, for each plane: point + normal
    in a numpy matrix. First row = 1st point (x,y,z), Second row = 1st normal (x,y,z). Then continue with the rest of PVs and LAA
    """
    clip_planes = np.zeros((10,3))
    regionslabels = getregionslabels()
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()

    # extract the body from the surface
    autolabels_full = surface.GetPointData().GetArray('autolabels')
    body = pointthreshold(surface, 'autolabels', regionslabels['body'], regionslabels['body'], 1)
    parts, max_n_cells = extract_all_regions(body)
    for part in parts:
        if part.GetNumberOfCells()==max_n_cells:
            body = part
        else:
            best_reg = -1
            min_dist_region = 1000
            for k in range(1,6):
                if k == 5:
                    index = 'laa'
                else:
                    index = 'pv' + str(k)
                vein = pointthreshold(surface, 'autolabels', regionslabels[index], regionslabels[index], 0)
                reg_dist, _, _ = min_euclidean_distance_fast(vtk_to_numpy(vein.GetPoints().GetData()), vtk_to_numpy(part.GetPoints().GetData()))
                if reg_dist<min_dist_region:
                    min_dist_region = reg_dist
                    best_reg=regionslabels[index]
            for i in range(part.GetNumberOfPoints()):
                point = locator.FindClosestPoint(part.GetPoint(i))
                autolabels_full.SetValue(point, best_reg)
    surface.GetPointData().AddArray(autolabels_full)
    writevtp(surface, ifile_sufix + 'autolabels.vtp')
    # initialize appender with the body
    appender = vtk.vtkAppendPolyData()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        appender.AddInputData(body)
    else:
        appender.AddInput(body)
    originaldist = targetdistance
    for k in range(1, 6):
        if k == 5:
            index = 'laa'
        else:
            index = 'pv' + str(k)
        # extract vein
        # excluding some points (alloff=0)
        # to avoid overlapping edges after appending
        vein = pointthreshold(surface, 'autolabels', regionslabels[index], regionslabels[index], 0)

        # load the centreline and the clipoint
        cl = readvtp(ifile_sufix + 'clvein' + str(k) + '.vtp')
        clippointid = int(np.loadtxt(ifile_sufix + 'clippointid' + str(k) + '.csv'))

        clippoint0 = cl.GetPoint(clippointid)
        clipnormal = (np.array(cl.GetPoint(clippointid + 1)) - np.array(cl.GetPoint(clippointid )))

        abscissasarray = cl.GetPointData().GetArray('Abscissas')
        startabscissa = abscissasarray.GetValue(clippointid)
        currentabscissa = 0
        currentid = clippointid

        # if different distance for 1 vein
        if specialvein > 0:
            if regionslabels[index] == specialvein:
                targetdistance = specialdist
            else:
                targetdistance = originaldist

        # find clip point
        while ((currentabscissa < targetdistance) and (currentabscissa > 0) and (currentid > 0)):
            currentid -= 1
            currentabscissa = startabscissa - abscissasarray.GetValue(currentid)

        if currentid > 0:
            currentid = currentid + 1
        else:
            # vein ended before target distance
            # then clip 4 mm before end of centreline from end point
            currentid = 4

        # clip and append
        clippoint1 = cl.GetPoint(currentid)
        clippedvein = planeclip(vein, clippoint1, clipnormal, 0)

        clip_planes[2*(k-1), 0:3] = clippoint1
        clip_planes[2*(k-1) +1, 0:3] = clipnormal

        # keep region closest to ostium point
        clippedvein = extractclosestpointregion(clippedvein, clippoint0)

        # clip generates new points to make a flat cut. The values may be interpolated.
        # we want all values to rounded to a certain label value.
        clippedvein = roundpointarray(clippedvein, 'autolabels')
        if vtk.vtkVersion.GetVTKMajorVersion() > 5:
            appender.AddInputData(clippedvein)
        else:
            appender.AddInput(clippedvein)

    # collect body + veins
    appender.Update()
    clippedsurface = appender.GetOutput()
    clippedsurface = cleanpolydata(clippedsurface)
    return clippedsurface, clip_planes

def find_mesh_intersection_along_vector(surface, start_point, direction, length=200.0):
    """
    Cast a line in both directions from the start_point along 'direction',
    and return the first intersection with the surface.
    """
    direction = direction / np.linalg.norm(direction)
    p0 = start_point - length * direction
    p1 = start_point + length * direction

    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(surface)
    obbTree.BuildLocator()

    points = vtk.vtkPoints()
    obbTree.IntersectWithLine(p0, p1, points, None)

    if points.GetNumberOfPoints() == 0:
        raise ValueError("[ERROR] No intersection found along direction. Check orientation or ray length.")

    intersection = [0.0, 0.0, 0.0]
    points.GetPoint(0, intersection)
    return np.array(intersection)

def estimate_mv_radius_from_surface(surface, max_hole_size=1000.0):
    """
    Estimate mitral valve diameter (in VTK units) from a surface with holes using published formula.
    Assumes the surface is in mm, and returns the diameter in mm.
    """

    import vtk
    import numpy as np 

    # Step 1: Fill holes to close the atrial body
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(surface)
    filler.SetHoleSize(max_hole_size)
    filler.Update()
    closed_surface = filler.GetOutput() 

    # Step 2: Compute volume (in mm^3)
    mass = vtk.vtkMassProperties()
    mass.SetInputData(closed_surface)
    mass.Update() 
    volume_mm3 = mass.GetVolume()

    if volume_mm3 <= 0:
        raise ValueError("Volume could not be computed — check surface quality.")

    # Step 3: Convert volume to mL
    # volume_ml = max(volume_mm3 / 1000.0, 100)  # since 1 mL = 1000 mm³
    volume_ml = max(volume_mm3 / 1000.0, 100)    
    print(f"\n\n [DEBUG] ESTIMATED VOLUME OF LA: {(volume_mm3/1000.0):.2f} mL")
    print(f"[DEBUG] TAKEN VOLUME OF LA: {volume_ml:.2f} mL")

    # Step 4: Estimate mitral annulus area (from publication formula)
    #ma_area_cm2 = (volume_ml - 50.0) / 12.0
    ma_area_cm2 = (volume_ml) / 12.0
    if ma_area_cm2 <= 0:
        raise ValueError("Estimated mitral area is non-positive — volume too small?")

    # Step 5: Convert area to diameter assuming circular annulus
    radius_cm = np.sqrt(ma_area_cm2 / np.pi)

    return radius_cm * 10.0  # return radius in mm


def find_mitral_sphere_pvs(surface, arrayname, outfile, eams=False, vis=0):
    """
    New version: Compute MV clipping sphere by combining body-to-ostia and body-to-leftPV directions.
    """
    
    # ========== Extract atrial body ==========
    border_edges = extractboundaryedge(surface)
    surface_filled = fillholes(surface, 1000) if border_edges.GetNumberOfPoints() > 0 else surface
    center_body = np.array(pointset_centreofmass(surface_filled))

    # ========== Extract ostia centers ==========
    if vis == 1:
        visualise_two_meshes(surface_filled, surface, arrayname)
    laa_ostium = pointthreshold(surface, arrayname, 36.5, 37.5)
    center_laa = np.array(pointset_centreofmass(laa_ostium))

    left_ostium = pointthreshold(surface_filled, arrayname, 78.0, 79.0)
    center_left_pv = np.array(centroidofcentroids(extractboundaryedge(left_ostium)))

    right_ostium = pointthreshold(surface_filled, arrayname, 76.0, 77.0)
    center_right_pv = np.array(centroidofcentroids(extractboundaryedge(right_ostium)))

    ripv_ostium = pointthreshold(surface_filled, arrayname, 75.0, 76.0)
    center_ripv = np.array(centroidofcentroids(extractboundaryedge(ripv_ostium)))

    lipv_ostium = pointthreshold(surface_filled, arrayname, 77.0, 78.0)
    center_lipv = np.array(centroidofcentroids(extractboundaryedge(lipv_ostium)))

    if not eams:
        # ========== Compute mean ostia centroid ==========
        center_ostia = (center_laa + center_left_pv + center_right_pv + center_ripv + center_lipv) / 5.0

        # ========== First direction: center_body → bottom of mesh ==========
        direction1 = normalizevector(center_ostia - center_body)
        bottom_point = find_mesh_intersection_along_vector(surface, center_body, direction1)

        # === Use midpoint instead of left PV directly ===
        midpoint_laa_left = (center_laa + center_left_pv) / 2

        # === Compute direction vectors ===
        vec_to_bottom = bottom_point - center_body
        vec_to_wall   = midpoint_laa_left - center_body
        vec_left_to_laa = center_laa - center_left_pv 

        # === Combine both directions and REVERSE (we want the other side!) ===
        combined_direction = -(6*vec_to_bottom + 1.5*vec_to_wall + 3*vec_left_to_laa)  # ← flip sign here!

        # === Compute the new clipping center on mesh surface ===
        clip_center = find_mesh_intersection_along_vector(surface, center_body, combined_direction)
        vectordown = multiplyvector(combined_direction / np.linalg.norm(combined_direction), -7)
        clip_center = acumvectors(vectordown, clip_center)
        # ========== Estimate MV radius from volume ==========
        radius_sphere = estimate_mv_radius_from_surface(surface)

        # ========== Perform sphere clipping ==========
        clipped_surface = sphereclip(surface, clip_center, radius_sphere)
        
    else:
        w = [0.95, -0.05, 0]
        scale=0.35
        # final pvscom average of left and right
        pvscom = acumvectors(center_left_pv, center_right_pv)
        pvscom = dividevector(pvscom, 2)

        # NOW AXES
        # Axis 1: Pvs com to body com
        pvdir = subtractvectors(center_body, pvscom)
        pvdirn = normalizevector(pvdir)

        # Axis 2: normal to Pvs axis
        ostiadir1 = subtractvectors(center_left_pv, center_right_pv)
        ostiadirn = normalizevector(ostiadir1)

        ostiacross = cross(pvdirn, ostiadirn)
        ostiacrossn = normalizevector(ostiacross)

        # Axis 3: normal to axis 1 and 2
        pvcross = cross(ostiacrossn, pvdirn)
        pvcrossn = normalizevector(pvcross)

        # thought of using for weighting but defualt values seem all right
        bodylength, pl1, pl2= computelengthalongvector(surface_filled, center_body, pvdirn)
        measurepoint = sumvectors(center_body, scale*bodylength, pvdirn)
        bodythick, pt1, pt2 = computelengthalongvector(surface_filled, measurepoint, ostiacrossn)
        bodywidth, pw1, pw2 = computelengthalongvector(surface_filled, measurepoint, pvcrossn)
        print('length', bodylength, 'width', bodywidth, 'thickness', bodythick)
        
        # clippoint with length vector
        length_start = euclideandistance(pl1, center_body)
        length_max = euclideandistance(pl1, center_body)
        loc_on_pw = normvector(np.array(measurepoint) - np.array(pw1))/ normvector(np.array(pw2) - np.array(pw1))
        linepts = pw1 + np.outer(np.linspace(min(loc_on_pw-0.05,0.45), max(loc_on_pw+0.05,0.55), 10), np.array(pw2) - np.array(pw1))
        if vis == 1:
            visualise_line_segment(surface_filled, center_body, measurepoint, pvdirn, pvcrossn, ostiacrossn, bodylength, bodywidth, bodythick, linepts[0], linepts[-1])
        for l, pt in enumerate(linepts):
            pvdir_temp = subtractvectors(pt, center_body)
            pvdirn_temp = normalizevector(pvdir_temp)
            _, pl1, _ = computelengthalongvector(surface_filled, center_body, pvdirn_temp)
            length_temp = euclideandistance(pl1, center_body)
            if length_temp>max(length_max, length_start+2.5):
                print('length increased by more than 2.5 mm, using different direction')
                length_max = length_temp
                pvdirn = pvdirn_temp
                # Axis 2: normal to Pvs axis
                ostiadir1 = subtractvectors(center_left_pv, center_right_pv)
                ostiadirn = normalizevector(ostiadir1)

                ostiacross = cross(pvdirn, ostiadirn)
                ostiacrossn = normalizevector(ostiacross)

                # Axis 3: normal to axis 1 and 2
                pvcross = cross(ostiacrossn, pvdirn)
                pvcrossn = normalizevector(pvcross)

                # thought of using for weighting but defualt values seem all right
                bodylength, pl1, pl2= computelengthalongvector(surface_filled, center_body, pvdirn)
                measurepoint = sumvectors(center_body, scale*bodylength, pvdirn)
                bodythick, pt1, pt2 = computelengthalongvector(surface_filled, measurepoint, ostiacrossn)
                bodywidth, pw1, pw2 = computelengthalongvector(surface_filled, measurepoint, pvcrossn)
                print(length_max - length_start)
                if (length_max - length_start) > 6:
                    print('length increased by  more than 6 mm, using different scale and weights') 
                    scale = 0.3
                    w = [0.9, -0.05, -0.2]
                print('length', bodylength, 'width', bodywidth, 'thickness', bodythick)
                if vis == 1:
                    visualise_body_dimensions_vtk(surface_filled, center_body, measurepoint, pvdirn, pvcrossn, ostiacrossn, bodylength, bodywidth, bodythick)
        
        if bodylength/bodywidth < 1.5:
            print("short body, using different scale and weights")
            scale = 0.3
            w = [0.95, 0.2, -0.2]

        pvdirnw = multiplyvector(pvdirn, w[0])
        ostiadirnw = multiplyvector(pvcrossn, w[1])
        ostiacrossnw = multiplyvector(ostiacrossn, w[2])

        plusvector = acumvectors(pvdirnw, ostiacrossnw)
        plusvector = acumvectors(plusvector, ostiadirnw)
        plusvectorn = normalizevector(plusvector)
        
        clippoint = sumvectors(center_body, scale*bodylength, pvdirn)

        vectordown = multiplyvector(plusvectorn, 20)
        clip_center = acumvectors(vectordown, clippoint)

        radius_sphere = estimate_mv_radius_from_surface(surface)

        clipped_surface = sphereclip(surface, clip_center, radius_sphere)
        if vis >= 1:
            visualise_mv_sphere_vtk(
                surface,
                center_body=center_body,
                center_left_pv=center_left_pv,
                center_laa=center_laa,
                clip_center=clip_center,
                direction_combined=vectordown,
                radius=radius_sphere
            )
    return clipped_surface

def select_mv_reference_point(surface, outfile):
    """
    Allows the user to select 1 interactive MV reference point for stable MV clipping.
    """
    print("Please click on the MV reference point (1 point only).")
    seeds = seed_interactor(surface, vtk.vtkPoints())

    if seeds.GetNumberOfIds() != 1:
        print("You must select exactly 1 point. Please try again.")
        return select_mv_reference_point(surface, outfile)

    # Extract the point coordinates
    mv_point = surface.GetPoint(seeds.GetId(0))
    newpoints = vtk.vtkPoints()
    newvertices = vtk.vtkCellArray()

    pid = newpoints.InsertNextPoint(mv_point)
    
    # Create the topology of the point (a vertex)
    newvertices.InsertNextCell(1)
    newvertices.InsertCellPoint(pid)
    pointspd = vtk.vtkPolyData()
    pointspd.SetPoints(newpoints)
    pointspd.SetVerts(newvertices)
    writevtp(pointspd, outfile)

    return np.array(mv_point)

def find_mitral_sphere_pvs_manual(surface, arrayname, outfile, seedfile, vis=0):
    """
    Detect the mitral valve region and clip the surface using a sphere centered along anatomical axis.
    """

    # ========== Extract atrial body ==========
    border_edges = extractboundaryedge(surface)
    surface_filled = fillholes(surface, 1000) if border_edges.GetNumberOfPoints() > 0 else surface
    body = pointthreshold(surface, arrayname, 36.0, 36.0)

    # ========== Compute anatomical centroids ==========
    center_body = np.array(pointset_centreofmass(body))
    laa_ostium = pointthreshold(surface, arrayname, 37.0, 37.0)
    center_laa = np.array(pointset_centreofmass(laa_ostium))
    left_ostium = pointthreshold(surface_filled, arrayname, 78.0, 79.0)
    center_left_pv = np.array(centroidofcentroids(extractboundaryedge(left_ostium)))
    right_ostium = pointthreshold(surface_filled, arrayname, 76.0, 77.0)
    center_right_pv = np.array(centroidofcentroids(extractboundaryedge(right_ostium)))
    center_pvs = (center_left_pv + center_right_pv) / 2
    midpoint_laa_left = (center_laa + center_left_pv) / 2

    # ========== Define local axes ==========
    axis_pvs_to_body = normalizevector(center_body - center_pvs)
    axis_left_to_right = normalizevector(center_left_pv - center_right_pv)
    axis_cross = normalizevector(np.cross(axis_pvs_to_body, axis_left_to_right))
    axis_third = normalizevector(np.cross(axis_cross, axis_pvs_to_body))

    # ========== Clipping axis ==========
    if not os.path.exists(seedfile):
        clippoint = select_mv_reference_point(surface, seedfile)
    else:
        print("Manual MV seedfile already exists. If you want to pick a new seed, remove the file ending with mvseed.vtp first.")
        pointspd = readvtp(seedfile)
        clippoint = pointspd.GetPoints().GetPoint(0)
    
    # ========== Estimate MV diameter from volume ==========
    radius_sphere = estimate_mv_radius_from_surface(surface)
    
    # ========== Perform sphere clipping ==========
    clipped_surface = sphereclip(surface, clippoint, radius_sphere)

    # ========== Optional visualization ==========
    if vis == 2:
        visualise_mv_sphere_vedo(
            surface, center_body, center_left_pv, center_laa,
            shifted_clippoint, clip_axis, radius_sphere,
            outfile
        )

    return clipped_surface
