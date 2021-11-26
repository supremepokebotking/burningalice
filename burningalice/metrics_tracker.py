import json
import enum

class METRIC_TYPES(enum.Enum):
    INCREMENT = 1
    MAX = 2
    APPEND = 3
    ADD_IF_DOESNT_EXIST = 4
    SET = 5
    SET_IF_NOT_NEGATIVE = 6
    ATLEAST_ONCE = 7
    MIN = 8

class MetricsTracker():
    def __init__(self, default_metrics):
        self.tracking_metrics = default_metrics
        self.metrics_this_turn = {}

        try:
            json.dumps(default_metrics)
        except Exception as e:
            print('default metrics is not serializable. if using a set, use list and use METRIC_TYPES.ADD_IF_DOESNT_EXIST to add items')
            raise e


    def bulk_update_metrics(self, metric_items):
        for prefix, metrics_key, value, metrics_type in metric_items:
            self.update_metrics(prefix, metrics_key, value, metrics_type)

    def update_metrics(self, prefix, metrics_key, value, metrics_type):
        full_key = metrics_key
        if metrics_key not in self.tracking_metrics.keys():
            full_key = '%s_%s' % (prefix, metrics_key)

#        print(full_key)
        if metrics_type == METRIC_TYPES.ATLEAST_ONCE:
            self.tracking_metrics[full_key] =  self.tracking_metrics[full_key] or value
        elif metrics_type == METRIC_TYPES.SET:
            self.tracking_metrics[full_key] = value
        elif metrics_type == METRIC_TYPES.MAX:
            self.tracking_metrics[full_key] = max(value, self.tracking_metrics[full_key])
        elif metrics_type == METRIC_TYPES.MIN:
            self.tracking_metrics[full_key] = min(value, self.tracking_metrics[full_key])
        elif metrics_type == METRIC_TYPES.APPEND:
            self.tracking_metrics[full_key].append(value)
        elif metrics_type == METRIC_TYPES.ADD_IF_DOESNT_EXIST:
            if value not in self.tracking_metrics[full_key]:
                self.tracking_metrics[full_key].append(value)
        elif metrics_type == METRIC_TYPES.INCREMENT:
            self.tracking_metrics[full_key] += 1
        elif metrics_type == METRIC_TYPES.SET_IF_NOT_NEGATIVE:
            if self.tracking_metrics[full_key] < 0:
                self.tracking_metrics[full_key] = value
        else:
            print('bad metrics_type', metrics_type)

        if full_key not in self.metrics_this_turn:
            self.metrics_this_turn[full_key] = []

        self.metrics_this_turn[full_key].append(self.tracking_metrics[full_key])


    def clear_metrics_this_turn(self):
        self.metrics_this_turn = {}
