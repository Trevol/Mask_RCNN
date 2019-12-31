from xml.etree import ElementTree as ET


class CvatAnnotation:
    class ImageAnnotation:
        def __init__(self, name, polygons=None, boxes=None):
            self.name = name
            self.polygons = [] if polygons is None else polygons
            self.boxes = [] if boxes is None else boxes

    class PolygonAnnotation:
        def __init__(self, label, points):
            self.label = label
            self.points = points

    class BoxAnnotation:
        def __init__(self, label, xtl, ytl, xbr, ybr):
            # <box label="pin-array" occluded="0" xtl="356.32" ytl="450.01" xbr="1587.91" ybr="877.64">
            pass

    @classmethod
    def parse(cls, annotationFile):
        tree = ET.parse(annotationFile)
        annotations = tree.getroot()

        labels = []
        for labelEl in annotations.find('meta').find('task').find('labels').findall('label'):
            labels.append(labelEl.find('name').text)

        imageAnnotations = []
        for imageEl in annotations.findall('image'):
            polygons = []
            for polygonEl in imageEl.findall('polygon'):
                label = polygonEl.get('label')
                points = polygonEl.get('points')
                points = _parsePoints(points)
                polygons.append(cls.PolygonAnnotation(label, points))

            boxes = []
            for boxEl in imageEl.findall('box'):
                label = boxEl.get('label')
                xtl = _floatStrToInt(boxEl.get('xtl'))
                ytl = _floatStrToInt(boxEl.get('ytl'))
                xbr = _floatStrToInt(boxEl.get('xbr'))
                ybr = _floatStrToInt(boxEl.get('ybr'))
                boxes.append(cls.BoxAnnotation(label, xtl, ytl, xbr, ybr))

            imageAnnotation = cls.ImageAnnotation(imageEl.get('name'), polygons, boxes)
            imageAnnotations.append(imageAnnotation)

        return labels, imageAnnotations


def _parsePoints(pointsString: str):
    # 546.47,391.58;555.35,383.94
    points = []
    for xy in pointsString.split(';'):
        x, y = xy.split(',')
        point = _floatStrToInt(x), _floatStrToInt(y)
        points.append(point)
    return points


def _floatStrToInt(strFloat):
    return int(round(float(strFloat)))


if __name__ == '__main__':
    labels, imageAnnotations = CvatAnnotation.parse('../tracking_arms_forceps/data/11_arm_forceps_solder_pin-array.xml')
    assert len(imageAnnotations) == 21
    assert imageAnnotations[0].polygons is not imageAnnotations[1].polygons
    assert imageAnnotations[0].boxes is not imageAnnotations[1].boxes
    assert len(imageAnnotations[3].polygons) == 2
    assert len(imageAnnotations[3].boxes) == 1
