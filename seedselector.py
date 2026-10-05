# From https://github.com/catactg/SUM

# adapted from VMTK centerlines
import vtk
from vmtk import vmtkrenderer

class vmtkSeedSelector(object):

    def __init__(self):
        self._Surface = None
        self._SeedIds = None
        self._SourceSeedIds = vtk.vtkIdList()

    def SetSurface(self, surface):
        self._Surface = surface

    def GetSurface(self):
        return self._Surface

    def GetSourceSeedIds(self):
        return self._SourceSeedIds

    def Execute(self):
        pass

class vmtkPickPointSeedSelector(vmtkSeedSelector):

    def __init__(self):
        vmtkSeedSelector.__init__(self)
        self.PickedSeedIds = vtk.vtkIdList()
        self.PickedSeeds = vtk.vtkPolyData()
        self.vmtkRenderer = None
        self.OwnRenderer = 0
        self.VisualizationPoints = vtk.vtkPoints()
        self.VisualizationActor = None

    def UndoCallback(self, obj):
        self.InitializeSeeds()
        self.PickedSeeds.Modified()
        self.vmtkRenderer.RenderWindow.Render()

    def PickCallback(self, obj):
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(1E-4 * self._Surface.GetLength())
        eventPosition = self.vmtkRenderer.RenderWindowInteractor.GetEventPosition()
        result = picker.Pick(float(eventPosition[0]),float(eventPosition[1]),0.0,self.vmtkRenderer.Renderer)
        if result == 0:
            return
        pickPosition = picker.GetPickPosition()
        pickedCellPointIds = self._Surface.GetCell(picker.GetCellId()).GetPointIds()
        minDistance = 1E10
        pickedSeedId = -1
        for i in range(pickedCellPointIds.GetNumberOfIds()):
            distance = vtk.vtkMath.Distance2BetweenPoints(pickPosition,self._Surface.GetPoint(pickedCellPointIds.GetId(i)))
            if distance < minDistance:
                minDistance = distance
                pickedSeedId = pickedCellPointIds.GetId(i)
        if pickedSeedId == -1:
            pickedSeedId = pickedCellPointIds.GetId(0)
        self.PickedSeedIds.InsertNextId(pickedSeedId)
        point = self._Surface.GetPoint(pickedSeedId)
        self.PickedSeeds.GetPoints().InsertNextPoint(point)
        self.PickedSeeds.Modified()
        self.vmtkRenderer.RenderWindow.Render()

    def InitializeSeeds(self):
        self.PickedSeedIds.Initialize()
        self.PickedSeeds.Initialize()
        seedPoints = vtk.vtkPoints()
        self.PickedSeeds.SetPoints(seedPoints)

    def SetVisualizationPoints(self, points):
        self.VisualizationPoints = points

    def VisualizePoints(self, points, color=(0.0, 1.0, 0.0)):
        """
        Visualize a vtkPoints object in the existing vmtk renderer.

        Parameters
        ----------
        points : vtk.vtkPoints
            Points to visualize.
        color : tuple
            RGB color, values from 0 to 1.
        scale_factor : float or None
            Sphere radius. If None, uses 1% of surface length.
        """

        point_polydata = vtk.vtkPolyData()
        point_polydata.SetPoints(points)

        glyph_source = vtk.vtkSphereSource()
        glyph_source.SetThetaResolution(12)
        glyph_source.SetPhiResolution(12)

        glyphs = vtk.vtkGlyph3D()

        if vtk.vtkVersion.GetVTKMajorVersion() > 5:
            glyphs.SetInputData(point_polydata)
        else:
            glyphs.SetInput(point_polydata)

        glyphs.SetSourceConnection(glyph_source.GetOutputPort())
        glyphs.SetScaleModeToDataScalingOff()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyphs.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.PickableOff()

        self.vmtkRenderer.Renderer.AddActor(actor)

    def Execute(self):

        self._SourceSeedIds.Initialize()

        if not self.vmtkRenderer:
            self.vmtkRenderer = vmtkrenderer.vmtkRenderer()
            self.vmtkRenderer.Initialize()
            self.OwnRenderer = 1
        
        if self.VisualizationPoints.GetNumberOfPoints() > 0:
            self.VisualizePoints(
                self.VisualizationPoints,
                color=(1.0, 165/255, 0.0)
            )

        glyphs = vtk.vtkGlyph3D()
        glyphSource = vtk.vtkSphereSource()
        if vtk.vtkVersion.GetVTKMajorVersion() > 5:
            glyphs.SetInputData(self.PickedSeeds)
        else:
            glyphs.SetInput(self.PickedSeeds)


        # if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        #     glyphs.SetSourceData(glyphSource.GetOutput())
        # else:
        #     glyphs.SetSource(glyphSource.GetOutput())
        if vtk.vtkVersion.GetVTKMajorVersion() > 5:
            glyphs.SetSourceConnection(glyphSource.GetOutputPort())
        else:
            glyphs.SetSourceConnection(glyphSource.GetOutputPort())

        glyphs.SetScaleModeToDataScalingOff()
        glyphs.SetScaleFactor(self._Surface.GetLength()*0.01)
        glyphMapper = vtk.vtkPolyDataMapper()


        # if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        #     glyphMapper.SetInputData(glyphs.GetOutput())
        # else:
        #     glyphMapper.SetInput(glyphs.GetOutput())
        glyphMapper.SetInputConnection(glyphs.GetOutputPort())


        self.SeedActor = vtk.vtkActor()
        self.SeedActor.SetMapper(glyphMapper)
        self.SeedActor.GetProperty().SetColor(1.0,0.0,0.0)
        self.SeedActor.PickableOff()
        self.vmtkRenderer.Renderer.AddActor(self.SeedActor)

        self.vmtkRenderer.AddKeyBinding('u','Undo.',self.UndoCallback)
        self.vmtkRenderer.AddKeyBinding('space','Add points.',self.PickCallback)
        surfaceMapper = vtk.vtkPolyDataMapper()
        if vtk.vtkVersion.GetVTKMajorVersion() > 5:
            surfaceMapper.SetInputData(self._Surface)
        else:
            surfaceMapper.SetInput(self._Surface)

        surfaceMapper.ScalarVisibilityOff()
        surfaceActor = vtk.vtkActor()
        surfaceActor.SetMapper(surfaceMapper)
        surfaceActor.GetProperty().SetOpacity(1)

        self.vmtkRenderer.Renderer.AddActor(surfaceActor)

#         # create a text actor
#         txt = vtk.vtkTextActor()
#         info = "Position mouse and press space. "
#         info += "Select seeds in this order: RSpv, RIpv, LIpv, LSpv. "
#         info += "Fifth seed requires --use_laa_seed command."
#         txt.SetInput(info)
#         txtprop=txt.GetTextProperty()
#         txtprop.SetFontFamilyToArial()
#         txtprop.SetFontSize(13)
#         txtprop.SetColor(1, 1, 1)
#         txt.SetDisplayPosition(0, 10)
#         self.vmtkRenderer.Renderer.AddActor(txt)
#
        any = 0
        while any == 0:
            self.InitializeSeeds()
            self.vmtkRenderer.Render()
            any = self.PickedSeedIds.GetNumberOfIds()
        self._SourceSeedIds.DeepCopy(self.PickedSeedIds)

        if self.OwnRenderer:
            self.vmtkRenderer.Deallocate()