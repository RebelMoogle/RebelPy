# VectorField module, supports Analytic and Griddata vectorfields. 

import numpy as np

class VectorField:
    """A class for handling 3D, time based Vector Fields.
       Uses numpy ndarrays for data. """
    _FlowData = 0

    def GetDataAt(x,y,z,t):
        return _FlowData[x,y,z,t]

    def __init__(self, FlowData, fieldName, **kwargs):
        import types
        """define VectorField with an Analytic Equation (function returning 3d vector) 
           or VectorFieldData as an ndarray,
           and a name of the field (string)"""
        if(isinstance(FlowData, types.FunctionType)): #AnalyticField with equation
            self.GetDataAt = FlowData
        elif(isinstance(FlowData, np.ndarray)):
            self._FlowData = FlowData
        else:
           raise Exception('INCORRECT PARAMETERS: please enter an analytic equation or an ndarray with data');
        self.fieldName = fieldName
        return super().__init__(**kwargs)


        


