import json
from .text_tracker import TextTracker
import cv2


DEFAULT_IGNORE_TRIGGER_THRESHOLDS = 0.05
DEFAULT_OVERRIDDE_RECT_THRESHOLDS = 0.93
DEFAULT_OCR_CONFIG =  {
    "always_use_list": False,
    "language_configs": [
        {
            "language": "eng",
            "use_easy_orc": False,
            "invert": False,
            "psm7": False,
        },
    ]
}


class TextConfigGenerator:

    def __init__(self, default_ocr_config=None):
        # only used for exluding from pairs
        self.labels_to_track = set()
        self.tracking_configurations = {}
        self.default_ocr_config = default_ocr_config

    def add_label_to_configuration(self, images, label_to_track, ocr_rect, other_ocr_rects=None, true_label=None, source_group=None, assisted_label_rect=None, ignore_if_labels_exists=None, ignore_if_regex_triggered=None, ignore_trigger_threshold=0.0, ignore_trigger_rects=None,
                override_rect_threshold=0.0, override_rects=None, ocr_config=None, other_ocr_configs=None, frames_to_skip=None, override_style=None):

        if true_label is None:
            true_label = label_to_track

        # true labels only. secondaries not included.
        self.labels_to_track.add(label_to_track)
        config = {
            "label": true_label,
            "position": "fixed",
            "ocr_rect": ocr_rect,
        }

        if frames_to_skip is None:
            frames_to_skip = 0

        if override_style in [None, 'base_style', 'custom']:
            config['override_style'] = override_style
        else:
            config['override_style'] = None

        config['frames_to_skip'] = frames_to_skip

        if other_ocr_rects is not None:
            config['other_ocr_rects'] = other_ocr_rects

        if assisted_label_rect is not None:
            config['assisted_label_rect'] = assisted_label_rect

        # add
        if source_group is not None:
            config['source_group'] = source_group

        if ignore_if_labels_exists is not None:
            config['ignore_if_labels_exists'] = ignore_if_labels_exists

        if ignore_if_regex_triggered is not None:
            config['ignore_if_regex_triggered'] = ignore_if_regex_triggered

        if ignore_trigger_rects is not None:
            ignore_config = {}
            ignore_config['rects'] = TextTracker.get_min_max_colors_for_regions_in_images(images, ignore_trigger_rects)
            ignore_config['threshold'] = ignore_trigger_threshold
            config['ignore_trigger_rects'] = ignore_config


        if override_rects is not None:
            config['override_rects'] = TextTracker.get_min_max_colors_for_regions_in_images(images, override_rects)
            config['threshold'] = override_rect_threshold

        if ocr_config is not None:
            config['ocr_config'] = ocr_config
        elif self.default_ocr_config is not None:
            config['ocr_config'] = self.default_ocr_config

        if other_ocr_configs is not None:
            config['other_ocr_configs'] = other_ocr_configs

        self.tracking_configurations[label_to_track] = config

    def get_generated_config(self):
        return {
            "labels_to_track": list(self.labels_to_track),
            "tracking_configurations": self.tracking_configurations,
        }

    def get_generated_config_as_json(self):
        return json.dumps(self.get_generated_config(), indent=4)







    @staticmethod
    def config_generator_from_mapping(mapping_json):
      label_keys = mapping_json["config_labels"]
      cfg_generator = TextConfigGenerator()

      for label_key in label_keys:
        label_to_track = label_key
        ocr_rect = mapping_json["ocr_rects"][label_key]
        assisted_label_rect = ocr_rect

        other_ocr_rects = None
        if "other_ocr_rects" in mapping_json:
          if label_to_track in mapping_json["other_ocr_rects"]:
            other_ocr_rects = mapping_json["other_ocr_rects"][label_to_track]

        if "assist_rects" in mapping_json:
          if label_to_track in mapping_json["assist_rects"]:
            assisted_label_rect = mapping_json["assist_rects"][label_to_track]

        ignore_trigger_threshold = DEFAULT_IGNORE_TRIGGER_THRESHOLDS
        override_rect_threshold = DEFAULT_OVERRIDDE_RECT_THRESHOLDS

        ignore_trigger_rects = None
        override_rects = None

        config_creation_images = []
        for image in mapping_json["images"][label_to_track]:
          if isinstance(image, str):
            config_creation_images.append(cv2.imread(image))
          else:
            config_creation_images.append(image)

        ocr_config = DEFAULT_OCR_CONFIG

        ignore_if_labels_exists = None
        ignore_if_regex_triggered = None

        true_label = label_to_track
        if "true_label_mapping" in mapping_json:
          if label_to_track in mapping_json["true_label_mapping"]:
            true_label = mapping_json["true_label_mapping"][label_to_track]


        source_group = None
        if "source_groups" in mapping_json:
          if label_to_track in mapping_json["source_groups"]:
            source_group = mapping_json["source_groups"][label_to_track]

        if "ignore_labels" in mapping_json:
          if label_to_track in mapping_json["ignore_labels"]:
            ignore_if_labels_exists = mapping_json["ignore_labels"][label_to_track]

        if "ignore_regexes" in mapping_json:
          if label_to_track in mapping_json["ignore_regexes"]:
            ignore_if_regex_triggered = mapping_json["ignore_regexes"][label_to_track]

        if "ignore_thresholds" in mapping_json:
          if label_to_track in mapping_json["ignore_thresholds"]:
            ignore_trigger_threshold = mapping_json["ignore_thresholds"][label_to_track]

        if "override_thresholds" in mapping_json:
          if label_to_track in mapping_json["override_thresholds"]:
            override_rect_threshold = mapping_json["override_thresholds"][label_to_track]

        if "override_rects" in mapping_json:
          if label_to_track in mapping_json["override_rects"]:
            override_rects = mapping_json["override_rects"][label_to_track]

        if "ignore_trigger_rects" in mapping_json:
          if label_to_track in mapping_json["ignore_trigger_rects"]:
            ignore_trigger_rects = mapping_json["ignore_trigger_rects"][label_to_track]


        if "ocr_configs" in mapping_json:
          if label_to_track in mapping_json["ocr_configs"]:
            ocr_config = mapping_json["ocr_configs"][label_to_track]

        other_ocr_configs = None
        if "other_ocr_configs" in mapping_json:
          if label_to_track in mapping_json["other_ocr_configs"]:
            other_ocr_configs = mapping_json["other_ocr_configs"][label_to_track]

        frames_to_skip = None
        if "frames_to_skip" in mapping_json:
          if label_to_track in mapping_json["frames_to_skip"]:
            frames_to_skip = mapping_json["frames_to_skip"][label_to_track]

        override_style = None
        if "override_styles" in mapping_json:
          if label_to_track in mapping_json["override_styles"]:
            override_style = mapping_json["override_styles"][label_to_track]

        cfg_generator.add_label_to_configuration(config_creation_images, label_to_track, ocr_rect, other_ocr_rects=other_ocr_rects, true_label=true_label, source_group=source_group, assisted_label_rect=assisted_label_rect, ignore_if_labels_exists=ignore_if_labels_exists,
                                              ignore_if_regex_triggered=ignore_if_regex_triggered,
                                              ignore_trigger_threshold=ignore_trigger_threshold, ignore_trigger_rects=ignore_trigger_rects,
                        override_rect_threshold=override_rect_threshold, override_rects=override_rects, ocr_config=ocr_config, other_ocr_configs=other_ocr_configs, frames_to_skip=frames_to_skip, override_style=override_style)
      return cfg_generator
