#Patent Pending

# Used to respond to events of text appearing on screen.
#    "apply_tesseract": False,

RECT_KEY = 'rect'
OCR_RECT_KEY = 'ocr_rect'
RECTS_KEY = 'rects'
ASSISTED_LABEL_RECT_KEY = 'assisted_label_rect'
THRESHOLD_KEY = 'threshold'
OCR_CONFIG_KEY = 'ocr_config'
LABELS_TO_TRACK_KEY = 'labels_to_track'
OVERRIDE_RECTS_KEY = 'override_rects'
IGNORE_IF_TRIGGERS_KEY = 'ignore_if_triggers'
IGNORE_IF_LABELS_EXIST_KEY = 'ignore_if_labels_exists'
IGNORE_IF_REGEX_TRIGGERED_KEY = 'ignore_if_regex_triggered'
REQUIRED_REGEX_TRIGGERED_KEY = 'ignore_if_regex_triggered'
TRACKING_CONFIGURATIONS_KEY = 'tracking_configurations'

import os
import re
REJECT_DUPLICATE_MESSAGES = bool(int(os.environ.get('REJECT_DUPLICATE_MESSAGES', 1)))
SAVE_MESSAGE_FRAMES = bool(int(os.environ.get('SAVE_MESSAGE_FRAMES', 1)))
ACCEPTED_MESSAGE_DIR = './accepted_message_frames'
REJECTED_MESSAGE_DIR = './rejected_message_frames'
DRAW_FRAMES_DIR = 'draw_frames'

import uuid

import io
from PIL import Image

try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3

class TextTracker():
    def __init__(self, config, callback=None, pool=None, custom_override_detector=None):

        self.config = config
        self.futures = []
        self.pool = pool
        self.callback = callback


        self.occurences  = {}
        self.assisted_occurences = {}
        self.ignored_occurences = {}
        self.processed_text = {}
        self.rejected_text = {}

        self.all_accepted_messages = []
        self.all_rejected_messages = []
        self.messages_in_order = []
        self.text_counts = {}

        # For local storage
        self.stored_messages_for_local = {}
        self.last_frame_mapping = {}
        self.messages_frame_count = {}
        self.current_frame_number = 0
        self.custom_override_detector = custom_override_detector

        for label_to_track in self.config[LABELS_TO_TRACK_KEY]:
            self.messages_frame_count[label_to_track] = 0

    def apply_async_pool(self, pool):
        self.pool = pool

    def apply_async_pool(self, pool):
        self.pool = pool

    def process_frame(self, frame_key, frame, label_and_boxes):

        labels_to_track = self.config[LABELS_TO_TRACK_KEY]
        # first pass is only for checking if frame exists or needs to be added.

        assist_labels = []

        # resetooo
        should_ignore_override_set = set()
        self.occurences  = {}
        self.assisted_occurences = {}
        self.ignored_occurences = {}
        self.processed_text = {}
        self.rejected_text = {}

        # used to know whether or not to consider variations
        # of the same label. pokemon_sword_message vs twitch_pokemon_sword
        # if one succeeds, the other doesnt need consideration
        processed_true_labels = set()

        active_source_group = None

        #this method will check pokemon_message, but not consider twitch_pokemon_message until override checks
        for label_to_track in labels_to_track:
            did_assist_in_detection = False

            if label_to_track not in label_and_boxes:
                continue

            text_track_configuration = self.config[TRACKING_CONFIGURATIONS_KEY][label_to_track]

            if "source_group" in text_track_configuration:
                active_source_group = text_track_configuration["source_group"]
                break


        # used to determine group before processing active

        for label_to_track in labels_to_track:
            did_assist_in_detection = False

            if label_to_track in label_and_boxes:
                continue
            if label_to_track not in self.config[TRACKING_CONFIGURATIONS_KEY]:
                print('unsupported label', label_to_track)
                continue

            text_track_configuration = self.config[TRACKING_CONFIGURATIONS_KEY][label_to_track]

            source_group = None

            if "source_group" in text_track_configuration:
                source_group = text_track_configuration["source_group"]

            # early exit for group barrier
            if source_group is not None and active_source_group is not None and source_group != active_source_group:
                continue


            text_track_rect = text_track_configuration[OCR_RECT_KEY]
            ocr_config = None
            if OCR_CONFIG_KEY in text_track_configuration:
                ocr_config = text_track_configuration[OCR_CONFIG_KEY]


            text_frame = self.get_subframe_from_frame(text_track_rect, frame)

            # check for ignore triggers
            if IGNORE_IF_TRIGGERS_KEY in text_track_configuration:
                override_data = text_track_configuration[IGNORE_IF_TRIGGERS_KEY]
                base_override_threshold = 0.05
                if THRESHOLD_KEY in override_data:
                    base_override_threshold = override_data[THRESHOLD_KEY]

                # check for overrides
                should_ignore_override, _ = TextTracker.text_override_in_frame(frame, override_data[RECTS_KEY], base_override_threshold)

                if should_ignore_override:
                    if frame_key not in self.ignored_occurences:
                        self.ignored_occurences[frame_key] = []

                    if label_to_track not in self.ignored_occurences[frame_key]:
                        self.ignored_occurences[frame_key].append(label_to_track)

                    if frame_key not in self.occurences:
                        self.occurences[frame_key] = []

                    if label_to_track not in self.text_counts:
                        self.text_counts[label_to_track] = 0

                    if label_to_track not in self.occurences[frame_key]:
                        self.occurences[frame_key].append(label_to_track)
                    self.text_counts[label_to_track] += 1

                    should_ignore_override_set.add(label_to_track)


            if OVERRIDE_RECTS_KEY not in text_track_configuration or len(text_track_configuration[OVERRIDE_RECTS_KEY]) == 0:
                continue

            override_style = None
            if "override_style" in text_track_configuration:
                override_style = text_track_configuration["override_style"]


            base_threshold = 0.95
            if THRESHOLD_KEY in text_track_configuration:
                base_threshold = text_track_configuration[THRESHOLD_KEY]
            base_threshold = 0.95

            did_assist_in_detection = False
            if override_style == "custom" and self.custom_override_detector is not None:
                did_assist_in_detection = self.custom_override_detector(frame, text_track_configuration[OVERRIDE_RECTS_KEY], base_threshold)

            else:
                # check for overrides
                override_for_text, _ = TextTracker.text_override_in_frame(frame, text_track_configuration[OVERRIDE_RECTS_KEY], base_threshold)

                #an assist did happen
                did_assist_in_detection = override_for_text

            if did_assist_in_detection:
                assist_labels.append(label_to_track)

                # add rect to labels and boxes.
                rect_for_assist = text_track_configuration[OCR_RECT_KEY]
                if ASSISTED_LABEL_RECT_KEY in text_track_configuration:
                    rect_for_assist = text_track_configuration[ASSISTED_LABEL_RECT_KEY]

                label_and_boxes[label_to_track] = [rect_for_assist]
                if frame_key not in self.assisted_occurences:
                    self.assisted_occurences[frame_key] = []

                if label_to_track not in self.assisted_occurences[frame_key]:
                    self.assisted_occurences[frame_key].append(label_to_track)

        # second pass
        for label_to_track in labels_to_track:

            text_track_configuration = self.config[TRACKING_CONFIGURATIONS_KEY][label_to_track]

            source_group = None

            if "source_group" in text_track_configuration:
                source_group = text_track_configuration["source_group"]

            if source_group is not None and active_source_group is not None and source_group != active_source_group:
                continue

            active_source_group = source_group

            true_tracking_label = text_track_configuration["label"]

            if true_tracking_label in processed_true_labels:
                continue

            if label_to_track not in label_and_boxes:
                continue

            if label_to_track in should_ignore_override_set:
                continue

            if label_to_track not in self.config[TRACKING_CONFIGURATIONS_KEY]:
                print('unsupported label', label_to_track)
                continue

            required_skip_frames = 0
            if "frames_to_skip" in text_track_configuration:
                required_skip_frames = text_track_configuration["frames_to_skip"]


            if self.messages_frame_count[label_to_track] == 0 or required_skip_frames == 0:
                pass
            elif self.current_frame_number - self.messages_frame_count[label_to_track] < required_skip_frames:
                continue

            self.messages_frame_count[label_to_track] = self.current_frame_number

            ignore_based_on_label = False
            if IGNORE_IF_LABELS_EXIST_KEY in text_track_configuration:
                for ignore_label in text_track_configuration[IGNORE_IF_LABELS_EXIST_KEY]:
                    if ignore_label in label_and_boxes:
                        ignore_based_on_label = True
            if ignore_based_on_label:
                if frame_key not in self.ignored_occurences:
                    self.ignored_occurences[frame_key] = []

                if label_to_track not in self.ignored_occurences[frame_key]:
                    self.ignored_occurences[frame_key].append(label_to_track)
                continue

            if frame_key not in self.occurences:
                self.occurences[frame_key] = []

            if label_to_track not in self.text_counts:
                self.text_counts[label_to_track] = 0

            if label_to_track not in self.occurences[frame_key]:
                self.occurences[frame_key].append(label_to_track)
            self.text_counts[label_to_track] += 1

            text_track_rect = text_track_configuration[OCR_RECT_KEY]

            ocr_config = {}
            if OCR_CONFIG_KEY in text_track_configuration:
                ocr_config = text_track_configuration[OCR_CONFIG_KEY]

            all_ocr_rects_and_configs = []
            all_ocr_rects_and_configs.append((None, text_track_rect, ocr_config))

            if "other_ocr_rects" in text_track_configuration:
                for ocr_key in text_track_configuration["other_ocr_rects"]:
                    ocr_config_to_use = ocr_config
                    if "other_ocr_configs" in text_track_configuration and ocr_key in text_track_configuration["other_ocr_configs"]:
                        ocr_config_to_use = text_track_configuration["other_ocr_configs"][ocr_key]
                    all_ocr_rects_and_configs.append((ocr_key, text_track_configuration["other_ocr_rects"][ocr_key], ocr_config))

            for ocr_rect_and_config in all_ocr_rects_and_configs:
                ocr_key, text_track_rect, ocr_config = ocr_rect_and_config


                message_frame = self.get_subframe_from_frame(text_track_rect, frame)
                retval, buffer = cv2.imencode('.png', message_frame)

                text_img_bytes = np.array(buffer).tostring()

                last_frame_key = '%s_%s' % (label_to_track, ocr_key)
                last_message_bytes = self.update_and_get_last_message_frame(text_img_bytes, last_frame_key)
                message_similarities = last_message_bytes is not None


                restricted_regex = None
                if IGNORE_IF_REGEX_TRIGGERED_KEY in text_track_configuration:
                    restricted_regex = text_track_configuration[IGNORE_IF_REGEX_TRIGGERED_KEY]

                if last_message_bytes is not None:
                    last_message_frame = np.array(Image.open(io.BytesIO(last_message_bytes)))

                    message_similarities = self.calculate_similar_frame_ratio(last_message_frame, message_frame)

                    if REJECT_DUPLICATE_MESSAGES and( last_message_bytes is not None and message_similarities >= 0.95):
                        print('Rejecting frame for duplicate message')
                        self.save_message_img(message_frame, accepted=False)
                    else:
                        self.save_message_img(message_frame, accepted=True)
                        if self.callback is not None:
                            self.callback(label_to_track, message_frame, ocr_config, ocr_key)
                        if self.pool is not None:
                            future = self.pool.apply_async(self.extract_english_japanese_text, [frame_key, message_frame, ocr_config, label_to_track, ocr_key, restricted_regex], callback=self.construct_pokemon_info_page_1)
                            self.futures.append(future)

                else:
                    self.save_message_img(message_frame, accepted=True)
                    if self.callback is not None:
                        self.callback(label_to_track, message_frame, ocr_config, ocr_key)
                    if self.pool is not None:
                        future = self.pool.apply_async(self.extract_english_japanese_text, [frame_key, message_frame, ocr_config, label_to_track, ocr_key, restricted_regex], callback=self.construct_pokemon_info_page_1)

        self.current_frame_number += 1
        return label_and_boxes, assist_labels

    def update_and_get_last_message_frame(self, current_frame, label):
        last_frame = None
        if label in self.last_frame_mapping:
            last_frame = self.last_frame_mapping[label]
        self.last_frame_mapping[label] = current_frame

        return last_frame

    def clear_last_message_frame(self, label):
        self.last_frame_mapping[label] = None

    # used for quick sanity checking
    def get_labels_and_ignore_override_ranges(self):
        labels_and_overrides = {}

        labels_to_track = self.config[LABELS_TO_TRACK_KEY]
        for label_to_track in labels_to_track:
            text_track_configuration = self.config[TRACKING_CONFIGURATIONS_KEY][label_to_track]

            # check for ignore triggers
            if IGNORE_IF_TRIGGERS_KEY not in text_track_configuration:
                continue

            override_configs = text_track_configuration[IGNORE_IF_TRIGGERS_KEY]
            max_red_gap = 0
            max_green_gap = 0
            max_blue_gap = 0

            labels_and_overrides[label_to_track] = {}
            labels_and_overrides[label_to_track]["rect_gaps"] = []

            for override_config in override_configs:
                red_gap = override_config["min_max_red"][1] - override_config["min_max_red"][0]
                green_gap = override_config["min_max_green"][1] - override_config["min_max_green"][0]
                blue_gap = override_config["min_max_blue"][1] - override_config["min_max_blue"][0]

                max_red_gap = max(max_red_gap, red_gap)
                max_green_gap = max(max_green_gap, green_gap)
                max_blue_gap = max(max_blue_gap, blue_gap)

                labels_and_overrides[label_to_track]["rect_gaps"].append((red_gap, green_gap, blue_gap))

            labels_and_overrides[label_to_track]["max_red_gap"] = max_red_gap
            labels_and_overrides[label_to_track]["max_green_gap"] = max_green_gap
            labels_and_overrides[label_to_track]["max_blue_gap"] = max_blue_gap

        return labels_and_overrides


    def get_labels_and_override_ranges(self):
        labels_and_overrides = {}

        labels_to_track = self.config[LABELS_TO_TRACK_KEY]
        for label_to_track in labels_to_track:

            text_track_configuration = self.config[TRACKING_CONFIGURATIONS_KEY][label_to_track]

            if OVERRIDE_RECTS_KEY not in text_track_configuration:
                continue

            override_configs = text_track_configuration[OVERRIDE_RECTS_KEY]
            max_red_gap = 0
            max_green_gap = 0
            max_blue_gap = 0

            labels_and_overrides[label_to_track] = {}
            labels_and_overrides[label_to_track]["rect_gaps"] = []

            for override_config in override_configs:
                red_gap = override_config["min_max_red"][1] - override_config["min_max_red"][0]
                green_gap = override_config["min_max_green"][1] - override_config["min_max_green"][0]
                blue_gap = override_config["min_max_blue"][1] - override_config["min_max_blue"][0]

                max_red_gap = max(max_red_gap, red_gap)
                max_green_gap = max(max_green_gap, green_gap)
                max_blue_gap = max(max_blue_gap, blue_gap)

                labels_and_overrides[label_to_track]["rect_gaps"].append((red_gap, green_gap, blue_gap))

            labels_and_overrides[label_to_track]["max_red_gap"] = max_red_gap
            labels_and_overrides[label_to_track]["max_green_gap"] = max_green_gap
            labels_and_overrides[label_to_track]["max_blue_gap"] = max_blue_gap

        return labels_and_overrides


    def get_warnings_for_overrides(self, threshold=100):
        ignored_warnings = []
        override_warnings = []

        ignored_labels_and_overrides = self.get_labels_and_ignore_override_ranges()
        labels_and_overrides = self.get_labels_and_override_ranges()

        for label in ignored_labels_and_overrides:
            override_data = ignored_labels_and_overrides[label]
            max_gap = max(override_data["max_red_gap"], override_data["max_red_gap"], override_data["max_red_gap"])
            if max_gap >= threshold:
                print('warning for: ', label, 'with max gap', max_gap, 'with data:', override_data)
                ignored_warnings.append((label, max_gap))

        for label in labels_and_overrides:
            override_data = labels_and_overrides[label]
            max_gap = max(override_data["max_red_gap"], override_data["max_red_gap"], override_data["max_red_gap"])
            if max_gap >= threshold:
                print('warning for: ', label, 'with max gap', max_gap, 'with data:', override_data)
                override_warnings.append((label, max_gap))

        return override_warnings, ignored_warnings

    @staticmethod
    def get_subframe_from_frame(rect, source_frame):
        h, w, c = source_frame.shape
        x1, y1, x2, y2 = rect
        x1 = min(max(x1, 0), w)
        x2 = min(max(x2, 0), w)
        y1 = min(max(y1, 0), h)
        y2 = min(max(y2, 0), h)
        crop_img = source_frame[y1:y2, x1:x2]
        return crop_img

    def add_messages_to_session(self, messages, label):
        if label not in self.stored_messages_for_local:
            self.stored_messages_for_local[label] = []

        self.stored_messages_for_local[label].append(messages)

    def get_messages_for_label(self, label):
        messages = []
        if label in self.stored_messages_for_local:
            messages = self.stored_messages_for_local[label].copy()
            self.stored_messages_for_local[label] = []
        return messages

    def any_messages_pending_for_label(self, label):
        if label not in self.stored_messages_for_local:
            return 0
        return len(self.stored_messages_for_local[label])

    @staticmethod
    def extract_english_japanese_text(frame_key, frame, ocr_config, label, ocr_key, regex=None):
#    def extract_english_japanese_text(self, frame):
        try:
            print('inside of extracto magneto')
            messages = PytesseractManager.process_image_for_text(frame, ocr_config)
        except Exception as e:
            print('Error occurred: %s ' % str(e))
            print(e)
            raise e
        return frame_key, messages, regex, label, ocr_key

    def construct_pokemon_info_page_1(self, future):
        frame_key, messages, regexes, label, ocr_key = future
        if frame_key not in self.processed_text:
            self.processed_text[frame_key] = {}

        if label not in self.processed_text[frame_key]:
            self.processed_text[frame_key][label] = []

        self.processed_text[frame_key][label].append(messages)

        self.messages_in_order.append((frame_key, label, messages))

        if regexes is None:
            self.add_messages_to_session(messages, label)
        else:
            for message in messages:
                for regex in regexes:
                    if re.search(regex, message):
                        if frame_key not in self.rejected_text:
                            self.rejected_text[frame_key] = {}

                        if label not in self.rejected_text[frame_key]:
                            self.rejected_text[frame_key][label] = []

                        self.rejected_text[frame_key][label].append(messages)
                        self.all_rejected_messages.append((frame_key, label, messages))
                        return
            self.add_messages_to_session(messages, label)
        self.all_accepted_messages.append((frame_key, label, messages))

        self.save_state(frame_key)


    @staticmethod
    def text_override_in_frame(img, rect_requirements, base_threshold):

        all_checks_pass = True
        failure_data = {}

        for requirement in rect_requirements:
            threshold = base_threshold
            if IGNORE_IF_TRIGGERS_KEY in requirement:
                threshold = requirement[THRESHOLD_KEY]

            crop_dimens = requirement[RECT_KEY]
            lower =( requirement["min_max_blue"][0], requirement["min_max_green"][0], requirement["min_max_red"][0]) # lower bound for each channel
            upper =( requirement["min_max_blue"][1], requirement["min_max_green"][1], requirement["min_max_red"][1]) # lower bound for each channel


            left = crop_dimens[0]
            top = crop_dimens[1]
            width = crop_dimens[2]
            height = crop_dimens[3]
            crop = img[top:height, left:width]

            all_checks_pass = all_checks_pass and (TextTracker.calculate_rect_ratio(crop, lower, upper) >= threshold)

            if not all_checks_pass:
                failure_data['crop_dimens'] = crop_dimens
                failure_data['requirement'] = requirement
                failure_data['percent'] = TextTracker.calculate_rect_ratio(crop, lower, upper)
                break

        return all_checks_pass, failure_data


    @staticmethod
    def calculate_rect_ratio(image, lower, upper):
        # make non grey pixels yellow
        img_adj = image.copy()

        # create the mask and use it to change the colors
        mask = cv2.inRange(img_adj, lower, upper)
        masked_color = [0,255,255]
        mask = 255 - mask
        img_adj[mask == 0] = masked_color
        return TextTracker.count_non_yellow_pixels(img_adj, masked_color)


    @staticmethod
    def count_non_yellow_pixels(image, mask):
        # grab the image dimensions
        h = image.shape[0]
        w = image.shape[1]
        target_pixels = 0


        # loop over the image, pixel by pixel
        for y in range(0, h):
            for x in range(0, w):
              pixel = image[y, x]
              if pixel[0] == mask[0] and pixel[1] == mask[1] and pixel[2] == mask[2]:
                target_pixels += 1
        return target_pixels / float(h*w)

    @staticmethod
    def calculate_similar_frame_ratio(imageData_1, imageData_2):

        valid_pixels_count = 0
        pixel_count = 0

        for i in range(0,imageData_1.shape[0]):
            for j in range(0,imageData_1.shape[1]):
                pixel_count += 1

                pixel_1 = [imageData_1.item(i, j, 0), imageData_1.item(i, j, 1), imageData_1.item(i, j, 2)]
                pixel_2 = [imageData_2.item(i, j, 0), imageData_2.item(i, j, 1), imageData_2.item(i, j, 2)]

                if pixel_1 == pixel_2:
                    valid_pixels_count += 1
        return valid_pixels_count / float(pixel_count)


    def save_message_img(self, message_frame, accepted):
        if not SAVE_MESSAGE_FRAMES:
            return

#        dir_to_use = '%s/%s' % (ACCEPTED_MESSAGE_DIR, self.state_manager.session_id)
        dir_to_use = ACCEPTED_MESSAGE_DIR
        if not accepted:
#            dir_to_use = '%s/%s' % (REJECTED_MESSAGE_DIR, self.state_manager.session_id)
            dir_to_use = REJECTED_MESSAGE_DIR

        if not os.path.exists(os.path.dirname(dir_to_use)):
            os.makedirs(os.path.dirname(dir_to_use))

        filename = '%s/%s.png' % (dir_to_use,  (str(uuid.uuid4())))
#        filename = '%s.png' % ((str(uuid.uuid4())))

        status = cv2.imwrite(filename, message_frame)


    @staticmethod
    def get_min_max_colors_for_regions_in_images(images, inspection_regions):

        result = []

        for crop_dimens in inspection_regions:
            left = crop_dimens[0]
            top = crop_dimens[1]
            width = crop_dimens[2]
            height = crop_dimens[3]

            black_min_red = 255
            black_max_red = 0
            black_min_green = 255
            black_max_green = 0
            black_min_blue = 255
            black_max_blue = 0

            for img in images:
                crop = img[top:height, left:width]

                # grab the image dimensions
                h = crop.shape[0]
                w = crop.shape[1]
                target_pixels = 0


                # loop over the image, pixel by pixel
                for y in range(0, h):
                    for x in range(0, w):

                        pixel = crop[y, x]
                        blue = pixel[0]
                        green = pixel[1]
                        red = pixel[2]

                        black_min_blue = int(min(black_min_blue, blue))
                        black_max_blue = int(max(black_max_blue, blue))
                        black_min_red = int(min(black_min_red, red))
                        black_max_red = int(max(black_max_red, red))
                        black_min_green = int(min(black_min_green, green))
                        black_max_green = int(max(black_max_green, green))

            result.append({
                "rect": crop_dimens,
                "min_max_red": [black_min_red, black_max_red],
                "min_max_green": [black_min_green, black_max_green],
                "min_max_blue": [black_min_blue, black_max_blue],
            })

        return result


    def compare_all_configs_againsts_other_images(self, config_mapper, restricted_labels = None):
        labels_to_track = self.config[LABELS_TO_TRACK_KEY]

        result = {}

        for label_to_track in labels_to_track:
            if restricted_labels is not None and label_to_track not in restricted_labels:
                continue

            tracking_configuration = self.config['tracking_configurations'][label_to_track]

            if 'override_rects' not in tracking_configuration:
                continue
            print("inspecting", label_to_track)

            threshold = tracking_configuration['threshold']

            other_images = set()

            for mapper_key in config_mapper["images"]:
                if mapper_key == label_to_track or config_mapper["images"][mapper_key] is None or len(config_mapper["images"][mapper_key]) == 0:
                    continue

                # removes lulua_item and lulua_atk combos
                for image_name in config_mapper["images"][mapper_key]:
                    if image_name not in config_mapper["images"][label_to_track]:
                        other_images.add(image_name)

            overriden_images = TextTracker.check_images_against_config_override(tracking_configuration, threshold, other_images)

            if len(overriden_images) > 0:
                result[label_to_track] = overriden_images

        return result

    # returns images that pass the override
    @staticmethod
    def check_images_against_config_override( tracking_configuration, threshold, image_names):
        overriden_images = []
        for image_name in image_names:
            image = cv2.imread(image_name)

            all_check_pass = True
            for rect_config in tracking_configuration['override_rects']:
                rect_image = TextTracker.get_subframe_from_frame(rect_config['rect'], image)
                # method expects list, since doing 1 at a time, force as list
                check_pass, failure_data = TextTracker.text_override_in_frame(image, [rect_config], threshold)
                all_check_pass = all_check_pass and check_pass
                if not all_check_pass:
                    break

            if all_check_pass:
                overriden_images.append(image_name)

        return sorted(overriden_images)


import json
import requests
import cv2
import numpy as np
import os


TESSERACT_BASE_URL = os.environ.get('TESSERACT_BASE_URL', 'http://localhost:5333/api/')
#TESSERACT_BASE_URL = os.environ.get('TESSERACT_BASE_URL', 'http://192.168.0.141:5333/api/')

class PytesseractManager():

    @staticmethod
    def fire_request( url, image, params):
        baseurl = TESSERACT_BASE_URL
        is_base64=False

        url = baseurl + url
        print('tesseract url:', url)

        if not is_base64:
#            headers = {'content-type': 'multipart/form-data'}

            _, image = cv2.imencode('.jpg', image)
            files = {
                'image': ('image', image, 'image/jpg'),
            }

            response = requests.post(url, data=params, files=files)

            return json.loads(response.text)['result']

        else:
            headers = {'content-type': 'application/json'}
            params['image'] =  image,

            response = requests.post(url, data=params, headers=headers)
            print('fire response', response)
            return json.loads(response.text)['result']

    @staticmethod
    def process_image_for_text( image, ocr_config=None):
        url = 'process_image_for_text'
        url = 'parse_rect_with_pytesseract_config'
        params = {}
        params['ocr_config'] = json.dumps(ocr_config)

        return PytesseractManager.fire_request(url, image, params)
