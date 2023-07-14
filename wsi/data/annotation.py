import json
import xml.etree.ElementTree as ET
import copy

import numpy as np
from skimage.measure import points_in_poly

np.random.seed(0)


class Polygon(object):

    def __init__(self, name, vertices):
        self._name = name
        self._vertices = vertices

    def __str__(self):
        return self._name

    def inside(self, coord):
        return points_in_poly([coord], self._vertices)[0]

    def vertices(self):

        return np.array(self._vertices)


class Annotation(object):
    def __init__(self):
        self._json_path = ''
        self._polygons_positive = []
        self._polygons_negative = []

    def __str__(self):
        return self._json_path

    def from_json(self, json_path):
        self._json_path = json_path
        with open(json_path) as f:
            annotations_json = json.load(f)

        for annotation in annotations_json['positive']:
            name = annotation['name']
            vertices = np.array(annotation['vertices'])
            polygon = Polygon(name, vertices)
            self._polygons_positive.append(polygon)

        for annotation in annotations_json['negative']:
            name = annotation['name']
            vertices = np.array(annotation['vertices'])
            polygon = Polygon(name, vertices)
            self._polygons_negative.append(polygon)

    def inside_polygons(self, coord, is_positive):
        if is_positive:
            polygons = copy.deepcopy(self._polygons_positive)
        else:
            polygons = copy.deepcopy(self._polygons_negative)

        return any(polygon.inside(coord) for polygon in polygons)

    def polygon_vertices(self, is_positive):
        if is_positive:
            return list(map(lambda x: x.vertices(), self._polygons_positive))
        else:
            return list(map(lambda x: x.vertices(), self._polygons_negative))


class Formatter(object):
    def camelyon16xml2json(self, outjson):
        root = ET.parse(self).getroot()
        annotations_tumor = \
                root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
        annotations_0 = \
                root.findall('./Annotations/Annotation[@PartOfGroup="_0"]')
        annotations_1 = \
                root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
        annotations_2 = \
                root.findall('./Annotations/Annotation[@PartOfGroup="_2"]')
        annotations_positive = \
                annotations_tumor + annotations_0 + annotations_1
        annotations_negative = annotations_2

        json_dict = {'positive': [], 'negative': []}
        for annotation in annotations_positive:
            X = list(map(lambda x: float(x.get('X')),
                     annotation.findall('./Coordinates/Coordinate')))
            Y = list(map(lambda x: float(x.get('Y')),
                     annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            name = annotation.attrib['Name']
            json_dict['positive'].append({'name': name, 'vertices': vertices})

        for annotation in annotations_negative:
            X = list(map(lambda x: float(x.get('X')),
                     annotation.findall('./Coordinates/Coordinate')))
            Y = list(map(lambda x: float(x.get('Y')),
                     annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            name = annotation.attrib['Name']
            json_dict['negative'].append({'name': name, 'vertices': vertices})

        with open(outjson, 'w') as f:
            json.dump(json_dict, f, indent=1)

    def vertices2json(self, positive_vertices=[], negative_vertices=[]):
        json_dict = {'positive': [], 'negative': []}
        for i in range(len(positive_vertices)):
            name = f'Annotation {i}'
            vertices = positive_vertices[i].astype(int).tolist()
            json_dict['positive'].append({'name': name, 'vertices': vertices})

        for i in range(len(negative_vertices)):
            name = f'Annotation {i}'
            vertices = negative_vertices[i].astype(int).tolist()
            json_dict['negative'].append({'name': name, 'vertices': vertices})

        with open(self, 'w') as f:
            json.dump(json_dict, f, indent=1)
