#!/usr/local/src/VTKBuild/bin/vtkpython
#!/opt/local/bin/vtkpython

import os
import sys
import os.path
sys.path.append('/usr/local/src/ParaView/lib/paraview-5.6/')
from vtk import *



import numpy as np
import vtk
from vtk.util import numpy_support as vtknp
import os
import sys
eps = sys.float_info.epsilon

def readvtup(input_file):
    
    reader = vtk.vtkUnstructuredGridReader()
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.SetFileName(input_file)
    reader.Update()
    output = reader.GetOutput()

    # print(output)

    NoP = output.GetNumberOfPoints()
    NoE = output.GetNumberOfCells()

    # coordinates of points
    temp_vtk_array = output.GetPoints().GetData()
    coord = vtknp.vtk_to_numpy(temp_vtk_array)

    conn = [ ]
    # cell connectivity
    for i in range(NoE):
        cell = output.GetCell(i)
        npts = cell.GetNumberOfPoints()
        connt = [cell.GetPointId(j) for j in range(npts)]
        conn.append(connt)

    conn = np.asarray(conn)

    return output, coord, conn


def getdata(dir):
    fileList=os.listdir(dir)
    fileList= [i for i in fileList if i.startswith('Body0') and i.endswith('14.vtk')]

    final_idx_y = 0
    final_min_y = 9999
    final_fname = ''

    volume_sum = 0

    for f in fileList:
        fileName=dir+'/'+f  

        output, coord, conn = readvtup(fileName)

        fields=output.GetCellData()
        mises=fields.GetArray("mises")
        mises = vtknp.vtk_to_numpy(mises)
        min_y = np.min(mises)
        idx_y = np.where(mises==np.min(mises))
        # print('fname: ', f, '   min_y: ', min_y, ' idx_y: ', idx_y)
        if(min_y < final_min_y):
            # print(max_y,final_max_y)
            final_min_y = min_y
            final_idx_y = idx_y
            final_fname = f

        fields=output.GetCellData()
        volume = fields.GetArray("volume")
        # print(volume)
        vloume = vtknp.vtk_to_numpy(volume)
        volume_sum = volume_sum + np.sum(volume)

    # print('final_fname: ', final_fname, '   final_min_y: ', final_min_y, ' final_idx_y: ', final_idx_y)
    print('sum of volume: ' ,volume_sum)

    return final_min_y,volume_sum


