import vtk
import numpy as np
import math
def euclideandistance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def furthest_point_to_polydata(pointset,refpoint):
    """Given set of points and ref point, select furthest point using euclidean dist"""
    refdist = 0
    for i in range(pointset.GetNumberOfPoints()):
        dist = euclideandistance(pointset.GetPoint(i),refpoint)
        if dist > refdist:
            refdist = dist
            selectedpointid = i
    return pointset.GetPoint(selectedpointid)

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

def visualise_default(surface, ref, case, arrayname, mini, maxi, points=None):
    """Visualise surface with a default parameters with colormap ranging from mini to maxi"""
    #Create a lookup table to map cell data to colors
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(255)
    lut.SetValueRange(0, 255)

    # qualitative data from colorbrewer  --> matching qualitative colormap of Paraview
    lut.SetTableValue(0, 0, 0, 0, 1)  #Black
    lut.SetTableValue(mini, 1, 1, 1, 1)   #white
    lut.SetTableValue(mini+1, 77/255.,175/255., 74/255., 1)   # green
    lut.SetTableValue(maxi-3, 152/255.,78/255.,163/255., 1)  # purple
    lut.SetTableValue(maxi-2, 255/255.,127/255., 0., 1)  # orange
    lut.SetTableValue(maxi-1, 55/255., 126/255., 184/255., 1)  # blue
    lut.SetTableValue(maxi, 166/255., 86/255., 40/255., 1)  # brown
    lut.Build()

    # create a text actor
    txt = vtk.vtkTextActor()
    txt.SetInput(case)
    txtprop=txt.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(18)
    txtprop.SetColor(0, 0, 0)
    txt.SetDisplayPosition(20, 30)

    # create a rendering window, renderer, and renderwindowinteractor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    # for GIMIAS interaction style
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(renWin)

    # surface mapper and actor
    surfacemapper = vtk.vtkPolyDataMapper()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        surfacemapper.SetInputData(surface)
    else:
        surfacemapper.SetInput(surface)
    surfacemapper.SetScalarModeToUsePointFieldData()
    surfacemapper.SelectColorArray(arrayname)
    surfacemapper.SetLookupTable(lut)
    surfacemapper.SetScalarRange(0,255)
    surfaceactor = vtk.vtkActor()
    # surfaceactor.GetProperty().SetOpacity(0)
    # surfaceactor.GetProperty().SetColor(1, 1, 1)
    surfaceactor.SetMapper(surfacemapper)

    pointmapper = vtk.vtkPolyDataMapper()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        pointmapper.SetInputData(points)
    else:
        pointmapper.SetInput(points)
    pointactor = vtk.vtkActor()
    # surfaceactor.GetProperty().SetOpacity(0)
    pointactor.GetProperty().SetColor(1, 1, 0)
    pointactor.GetProperty().SetPointSize(10)  
    pointactor.SetMapper(pointmapper)

    # refsurface mapper and actor
    refmapper = vtk.vtkPolyDataMapper()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        refmapper.SetInputData(ref)
    else:
        refmapper.SetInput(ref)
    refmapper.SetScalarModeToUsePointFieldData()
    refmapper.SelectColorArray(arrayname)
    refmapper.SetLookupTable(lut)
    refmapper.SetScalarRange(0,255)
    refactor = vtk.vtkActor()
    refactor.GetProperty().SetOpacity(0.5)
    # refactor.GetProperty().SetColor(1, 1, 1)
    refactor.SetMapper(refmapper)

    # assign actors to the renderer
    ren.AddActor(refactor)
    ren.AddActor(surfaceactor)
    ren.AddActor(txt)
    ren.AddActor(pointactor)
    # set the background and size; zoom in; and render
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(1280, 960)
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1)


    # enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()

    outcam = ren.GetActiveCamera()
    # print("after", outcam.GetViewUp())

def visualise_two_meshes(surface, ref, case=""):
    """Visualise surface in solid color and 'ref' in transparent"""
    # create a text actor
    txt = vtk.vtkTextActor()
    txt.SetInput(case)
    txtprop=txt.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(18)
    txtprop.SetColor(0, 0, 0)
    txt.SetDisplayPosition(20, 30)

    # create a rendering window, renderer, and renderwindowinteractor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    # for GIMIAS interaction style
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(renWin)

    # surface mapper and actor
    surfacemapper = vtk.vtkPolyDataMapper()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        surfacemapper.SetInputData(surface)
    else:
        surfacemapper.SetInput(surface)
    surfacemapper.SetScalarModeToUsePointFieldData()
    surfaceactor = vtk.vtkActor()
    # surfaceactor.GetProperty().SetOpacity(0)
    surfaceactor.GetProperty().SetColor(288/255, 26/255, 28/255)
    surfaceactor.SetMapper(surfacemapper)

    # refsurface mapper and actor
    refmapper = vtk.vtkPolyDataMapper()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        refmapper.SetInputData(ref)
    else:
        refmapper.SetInput(ref)
    refmapper.SetScalarModeToUsePointFieldData()

    refactor = vtk.vtkActor()
    refactor.GetProperty().SetOpacity(0.5)
    refactor.GetProperty().SetColor(1, 1, 1)
    refactor.SetMapper(refmapper)

    # assign actors to the renderer
    # ren.AddActor(refactor)
    ren.AddActor(surfaceactor)
    ren.AddActor(refactor)
    ren.AddActor(txt)

    # set the background and size; zoom in; and render
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(800, 800)
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1)

    # enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()

def visualise_veins(surface, surface2, ref, case=""):
    """Visualise surface in solid color and 'ref' in trasparent"""
    # create a text actor
    txt = vtk.vtkTextActor()
    txt.SetInput(case)
    txtprop=txt.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(18)
    txtprop.SetColor(0, 0, 0)
    txt.SetDisplayPosition(20, 30)

    # create a rendering window, renderer, and renderwindowinteractor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    # for GIMIAS interaction style
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(renWin)

    # surface mapper and actor
    surfacemapper = vtk.vtkPolyDataMapper()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        surfacemapper.SetInputData(surface)
    else:
        surfacemapper.SetInput(surface)
    surfacemapper.SetScalarModeToUsePointFieldData()
    surfaceactor = vtk.vtkActor()
    # surfaceactor.GetProperty().SetOpacity(0)
    surfaceactor.GetProperty().SetColor(28/255, 26/255, 288/255)
    surfaceactor.SetMapper(surfacemapper)

    # surface mapper and actor
    surfacemapper2 = vtk.vtkPolyDataMapper()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        surfacemapper2.SetInputData(surface2)
    else:
        surfacemapper2.SetInput(surface)
    surfacemapper2.SetScalarModeToUsePointFieldData()
    surfaceactor2 = vtk.vtkActor()
    # surfaceactor.GetProperty().SetOpacity(0)
    surfaceactor2.GetProperty().SetColor(288/255, 26/255, 28/255)
    surfaceactor2.SetMapper(surfacemapper2)

    # refsurface mapper and actor
    refmapper = vtk.vtkPolyDataMapper()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        refmapper.SetInputData(ref)
    else:
        refmapper.SetInput(ref)
    refmapper.SetScalarModeToUsePointFieldData()

    refactor = vtk.vtkActor()
    refactor.GetProperty().SetOpacity(0.5)
    refactor.GetProperty().SetColor(1, 1, 1)
    refactor.SetMapper(refmapper)

    # assign actors to the renderer
    # ren.AddActor(refactor)
    ren.AddActor(surfaceactor)
    ren.AddActor(surfaceactor2)
    ren.AddActor(refactor)
    ren.AddActor(txt)

    # set the background and size; zoom in; and render
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(800, 800)
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1)

    # enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()

def vis_paths(m_open, conts, paths):
    # Base mesh
    renderer.AddActor(add_actor(m_open, color=(0.8, 0.8, 0.8)))  # Gray base

    # Contours
    cont_colors = {
        "rspv": (1, 0, 0),   # Red
        "ripv": (0, 1, 0),   # Green
        "lipv": (0, 0, 1),   # Blue
        "lspv": (1, 1, 0),   # Yellow
        "mv":   (1, 0, 1),   # Magenta
        "laa":  (0, 1, 1)    # Cyan
    }
    for i, cont in enumerate(conts):
        print(list(cont_colors.values()))
        renderer.AddActor(add_actor(cont, list(cont_colors.values())[i], linewidth=3))
    

    # Paths
    for path in paths:
        i+=1
        renderer.AddActor(add_actor(path, list(cont_colors.values())[i], linewidth=2))  # Orange path

    # Camera and rendering
    renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background
    render_window.SetSize(800, 800)
    render_window.Render()
    interactor.Start()
import vtk
import numpy as np


def make_sphere_actor(center, radius, color, opacity=1.0, resolution=32):
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(center)
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(resolution)
    sphere.SetPhiResolution(resolution)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)

    return actor


def make_line_actor(p1, p2, color, width=2):
    line = vtk.vtkLineSource()
    line.SetPoint1(p1)
    line.SetPoint2(p2)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(width)

    return actor


def make_arrow_actor(start, end, color):
    arrow_source = vtk.vtkArrowSource()

    start = np.array(start)
    end = np.array(end)

    direction = end - start
    length = np.linalg.norm(direction)

    direction = direction / length

    arbitrary = np.array([1, 0, 0])
    if np.abs(np.dot(direction, arbitrary)) > 0.9:
        arbitrary = np.array([0, 1, 0])

    normal = np.cross(direction, arbitrary)
    normal /= np.linalg.norm(normal)

    binormal = np.cross(normal, direction)

    matrix = vtk.vtkMatrix4x4()

    matrix.SetElement(0, 0, direction[0])
    matrix.SetElement(1, 0, direction[1])
    matrix.SetElement(2, 0, direction[2])

    matrix.SetElement(0, 1, normal[0])
    matrix.SetElement(1, 1, normal[1])
    matrix.SetElement(2, 1, normal[2])

    matrix.SetElement(0, 2, binormal[0])
    matrix.SetElement(1, 2, binormal[1])
    matrix.SetElement(2, 2, binormal[2])

    transform = vtk.vtkTransform()
    transform.Translate(start)
    transform.Concatenate(matrix)
    transform.Scale(length, length, length)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputConnection(arrow_source.GetOutputPort())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(transform_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)

    return actor


def visualise_mv_sphere_vtk(
        surface,
        center_body,
        center_left_pv,
        center_laa,
        clip_center,
        direction_combined,
        radius=10.0):

    center_body = np.array(center_body)
    center_left_pv = np.array(center_left_pv)
    center_laa = np.array(center_laa)
    clip_center = np.array(clip_center)
    direction_combined = np.array(direction_combined)

    midpoint = (center_laa + center_left_pv) / 2

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1400, 1000)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Surface mesh
    poly = vtk.vtkPolyData()
    poly.DeepCopy(surface)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)

    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mapper)
    mesh_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
    mesh_actor.GetProperty().SetOpacity(0.3)

    renderer.AddActor(mesh_actor)

    # Points
    renderer.AddActor(make_sphere_actor(center_body, 0.8, (1, 0, 0)))
    renderer.AddActor(make_sphere_actor(center_laa, 0.6, (0, 0, 0)))
    renderer.AddActor(make_sphere_actor(center_left_pv, 0.6, (0, 0, 1)))
    renderer.AddActor(make_sphere_actor(midpoint, 0.6, (1, 0.84, 0)))
    renderer.AddActor(make_sphere_actor(clip_center, 0.8, (0.5, 0.5, 0.5)))

    # Body -> midpoint line
    renderer.AddActor(make_line_actor(center_body, midpoint, (1, 0.5, 0), 3))

    # Combined direction arrow
    renderer.AddActor(
        make_arrow_actor(
            center_body,
            center_body + direction_combined,
            (0, 0, 0)
        )
    )

    # Clipping sphere
    renderer.AddActor(
        make_sphere_actor(
            clip_center,
            radius,
            (0.5, 0.5, 0.5),
            opacity=0.4,
            resolution=64
        )
    )

    # Camera
    renderer.ResetCamera()

    render_window.Render()
    interactor.Start()

def make_line_actor(p1, p2, color, width=4):
    line = vtk.vtkLineSource()
    line.SetPoint1(p1)
    line.SetPoint2(p2)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(width)

    return actor

def visualise_computelengthalongvector(renderer,
        polydata,
        refpoint,
        vector, color):

    """
    Visualize computelengthalongvector():
    - reference point
    - measurement vector
    - infinite probing line
    - intersection points
    - final measured segment
    """

    refpoint = np.array(refpoint)
    vector = np.array(vector)

    vector = vector / np.linalg.norm(vector)

    # Long probing line
    point_forward = refpoint + 1000 * vector
    point_backward = refpoint - 1000 * vector

    # Intersections forward
    intersectpoints1 = intersectwithline(
        polydata,
        refpoint,
        point_forward
    )

    furthestpoint1 = furthest_point_to_polydata(
        intersectpoints1,
        refpoint
    )

    # Intersections backward
    intersectpoints2 = intersectwithline(
        polydata,
        refpoint,
        point_backward
    )

    furthestpoint2 = furthest_point_to_polydata(
        intersectpoints2,
        furthestpoint1
    )

    length = euclideandistance(
        furthestpoint1,
        furthestpoint2
    )
  
    # Final measured segment
    renderer.AddActor(
        make_line_actor(
            furthestpoint1,
            furthestpoint2,
            color,
            width=6
        )
    )

    # Endpoints
    renderer.AddActor(
        make_sphere_actor(
            furthestpoint1,
            1.2,
            color
        )
    )

    renderer.AddActor(
        make_sphere_actor(
            furthestpoint2,
            1.2,
            color
        )
    )


    # Print measured length
    print("Measured length:", length)


def visualise_body_dimensions_vtk(
        body,
        center_body,
        measurepoint,
        pvdirn,
        pvcrossn,
        ostiacrossn,
        bodylength,
        bodywidth,
        bodythick):

    """
    Visualize:
    - body length direction
    - body width
    - body thickness
    """

    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1400, 1000)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Surface mesh
    poly = vtk.vtkPolyData()
    poly.DeepCopy(body)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)

    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mapper)

    mesh_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
    mesh_actor.GetProperty().SetOpacity(0.3)

    renderer.AddActor(mesh_actor)

    # Main points
    visualise_computelengthalongvector(renderer, body, center_body, pvdirn, (0, 0, 1))
    visualise_computelengthalongvector(renderer, body, measurepoint, ostiacrossn, (0, 0.7, 0))
    visualise_computelengthalongvector(renderer, body, measurepoint, pvcrossn, (0.6, 0, 0.6))
    
    renderer.AddActor(
        make_sphere_actor(center_body, 1.0, (1, 0, 0))
    )

    renderer.AddActor(
        make_sphere_actor(measurepoint, 1.0, (0, 0, 0))
    )

    renderer.ResetCamera()

    render_window.Render()
    interactor.Start()

