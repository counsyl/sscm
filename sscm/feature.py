"""
Loads information from the features.json file
"""
import json

from parameter import Feature, MultiFeature, ModelFeature


class Features(object):

    def __init__(self, columns, feature_map):
        """
        columns: a list of names associated with each column in order
        feature_map: a dictionary mapping a feature name with
                     its Feature object
        """
        self.columns = columns
        self.feature_map = feature_map
        self.load_models()

    @staticmethod
    def load(feature_file, models_dir):
        """
        Creates a Features object from an
        appropriate JSON file (features_file).

        Format (with 3 different types of features):
            {
                "columns": [ "column_name1", "column_name2", ... ],
                "features": {
                    "feature_name1": {
                        "feature": "scalar",
                        "column": "column_name1",
                        "type": "float",
                        "distribution": "Gaussian"
                    },
                    "feature_name2": {
                        "feature": "scalar",
                        "column": "Consequence",
                        "type": "string",
                        "distribution": "Multinomial"
                    },
                    "feature_name3": {
                        "feature": "vector",
                        "columns": ["column_name1", "column_name2"],
                        "type": "float",
                        "distribution": "MultivariateGaussian"
                    },
                    ...
                }
            }

        """

        DATA_TYPE = {
            "float": float,
            "string": str,
            "int": int
        }

        def get_type(feature_type):
            return DATA_TYPE[feature_type]

        with open(feature_file, 'r') as fp:
            feature_json = json.load(fp)
            column_names = feature_json['columns']
            feature_map = {}
            for name, info in feature_json['features'].iteritems():
                params = info.get('info', {})
                feature_type = get_type(info['type'])
                distribution = info['distribution']
                preprocess = info.get('preprocess', {})
                if info['feature'] == 'scalar':
                    column = info['column']
                    feature = Feature(name,
                                      column_names.index(column),
                                      feature_type,
                                      distribution,
                                      preprocess,
                                      params)
                elif info['feature'] == 'vector':
                    columns = [column_names.index(n) for n in info['columns']]
                    feature = MultiFeature(name,
                                           info['columns'],
                                           columns,
                                           [feature_type] * len(columns),
                                           distribution,
                                           preprocess,
                                           params)
                elif info['feature'] == 'model':
                    source = info.get('source_cluster', 0)
                    feature = ModelFeature(name,
                                           feature_type,
                                           distribution,
                                           info['model_name'],
                                           models_dir,
                                           source,
                                           preprocess,
                                           params)
                feature_map[name] = feature
        fm = Features(column_names, feature_map)
        return fm

    def load_models(self):
        for name, feature in self.feature_map.items():
            if isinstance(feature, ModelFeature):
                feature.load(self)

    def get_feature(self, name):
        if name not in self.feature_map:
            raise Exception("Feature %s doesn't exist" % name)
        return self.feature_map[name]

    def get_index(self, name):
        return self.columns.index(name)
