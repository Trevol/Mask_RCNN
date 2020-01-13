from xml.etree import ElementTree as ET


def main():
    annotation = '23_vid6_ arm_forceps_solder_pin-array.xml'
    # sort nodes by id
    tree = ET.parse(annotation)
    annotations = tree.getroot()
    images: [ET.Element] = annotations.findall('image')
    images.sort(key=lambda im: im.get('id'))

    for im in images:
        annotations.remove(im)
    for im in images:
        annotations.append(im)

    tree.write('tmp.xml')


main()
