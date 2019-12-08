class CvatAnnotation:
    class ImageAnnotation:
        # __slots__ = ['id', 'name', 'polygons']
        def __init__(self):
            self.name = '11111'

    class PolygonAnnotation:
        # __slots__ = ['label', 'points']
        def __init__(self):
            pass

    @staticmethod
    def parse(annotationFile):
        pass
