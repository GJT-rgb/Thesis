import ezdxf

doc = ezdxf.readfile(r"E:/EMBL/Master EMBL/Python/polyline dxf/Cardio_projection3D.dxf")
msp = doc.modelspace()

entity_types = set(e.dxftype() for e in msp)

print("DXF contains these entity types:", entity_types)
