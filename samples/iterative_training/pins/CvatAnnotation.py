import xml.etree.ElementTree as ET


class CvatAnnotation:
    class ImageAnnotation:
        def __init__(self, name, polygons=None):
            self.name = name
            self.polygons = [] if polygons is None else polygons

    class PolygonAnnotation:
        def __init__(self, label, points):
            self.label = label
            self.points = points

    @staticmethod
    def _parsePoints(pointsString: str):
        # 546.47,391.58;555.35,383.94
        points = []
        for xy in pointsString.split(';'):
            x, y = xy.split(',')
            point = int(round(float(x))), int(round(float(y)))
            points.append(point)
        return points

    @classmethod
    def parse(cls, annotationFile):
        tree = ET.parse(annotationFile)
        annotations = tree.getroot()

        labels = []
        for labelEl in annotations.find('meta').find('task').find('labels').findall('label'):
            labels.append(labelEl.find('name').text)

        imageAnnotations = []
        for imageEl in annotations.findall('image'):
            imagePolygonAnnotations = []
            for polygonEl in imageEl.findall('polygon'):
                label = polygonEl.get('label')
                points = polygonEl.get('points')
                points = cls._parsePoints(points)
                polygonAnnotation = CvatAnnotation.PolygonAnnotation(label, points)
                imagePolygonAnnotations.append(polygonAnnotation)
            imageAnnotation = CvatAnnotation.ImageAnnotation(imageEl.get('name'), imagePolygonAnnotations)
            imageAnnotations.append(imageAnnotation)

        return labels, imageAnnotations


if __name__ == '__main__':
    labels, imageAnnotations = CvatAnnotation.parse('1_TestSegmentation.xml')
    assert len(imageAnnotations) == 5
    assert imageAnnotations[0].polygons is not imageAnnotations[1].polygons
    assert len(imageAnnotations[0].polygons) == 6
    assert len(imageAnnotations[1].polygons) == 12
