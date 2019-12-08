import xml.etree.ElementTree as ET


class CvatAnnotation:
    class ImageAnnotation:
        def __init__(self, name, polygons=[]):
            self.name = name
            self.polygons = polygons

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
            imageAnnotation = CvatAnnotation.ImageAnnotation(imageEl.get('name'))
            imageAnnotations.append(imageAnnotation)
            for polygonEl in imageEl.findall('polygon'):
                label = polygonEl.get('label')
                points = polygonEl.get('points')
                points = cls._parsePoints(points)
                polygonAnnotation = CvatAnnotation.PolygonAnnotation(label, points)
                imageAnnotation.polygons.append(polygonAnnotation)

        return labels, imageAnnotations


if __name__ == '__main__':
    CvatAnnotation.parse('1_TestSegmentation.xml')
