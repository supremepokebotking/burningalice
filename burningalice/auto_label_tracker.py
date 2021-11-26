import cv2

stats_json = {
    'frame_count': 1,
    'assisted_count': 12,
    'assisted_filenames': [],
    'seen_labels': [],
    'assisted_labels': [],
    'filenames': [],
}

config_json = {
    'default_use_centroid': False,
    'centroid_labels': [],
    'rect_labels': [],
    'rect_percentage_threshold': 0.8,
    'centroid_distance_treshold': 50,
}

# raw image
IMAGES_DIR = './images'
#images with boxes
LABELED_IMAGES_DIR = './labeled_images'
# yolo txt files
YOLO_LABEL_DIR = './yolo'
#where metrics is
METRICS_DIR = './metrics'

import csv

import numpy as np
import json

import os
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

if not os.path.exists(LABELED_IMAGES_DIR):
    os.makedirs(LABELED_IMAGES_DIR)

if not os.path.exists(YOLO_LABEL_DIR):
    os.makedirs(YOLO_LABEL_DIR)

if not os.path.exists(METRICS_DIR):
    os.makedirs(METRICS_DIR)

class AutoLabelTracker():
    def __init__(self, config, all_label_names, session_id):
        self.session_id = session_id
        self.config = config
        self.all_label_names  = all_label_names
        self.colors  = self.get_colors(all_label_names)

        self.first_of_kinds = []
        self.second_of_kinds = []

        self.frame_count = 0
        self.json_generator = []
        self.video_assets = {}
        self.image_asset_data = []
        self.last_label_keys = None

        self.frames_and_label_and_boxes = {}

        # For the sake of being about to update labels and boxes at a later time
        # and not having to loop through every source to fix reference
        self.labels_and_boxes_key_mapping = {}
        self.labels_to_ignore_during_comparison = []

        self.images_session_dir = '%s/%s' % (IMAGES_DIR, self.session_id)
        self.labeled_images_session_dir = '%s/%s' % (LABELED_IMAGES_DIR, self.session_id)
        self.yolo_label_session_dir = '%s/%s' % (YOLO_LABEL_DIR, self.session_id)
        self.metrics_session_dir = '%s/%s' % (METRICS_DIR, self.session_id)

    def create_session_folders(self):
        if not os.path.exists(self.images_session_dir):
            os.makedirs(self.images_session_dir)

        if not os.path.exists(self.labeled_images_session_dir):
            os.makedirs(self.labeled_images_session_dir)

        if not os.path.exists(self.yolo_label_session_dir):
            os.makedirs(self.yolo_label_session_dir)

        if not os.path.exists(self.metrics_session_dir):
            os.makedirs(self.metrics_session_dir)

    def is_same(self, label_and_boxes, other_label_and_boxes):
        rect_overlap_percent = 0.7
        label_keys = label_and_boxes.keys()
        other_label_keys = other_label_and_boxes.keys()

        # remove item from keys if in labels  blah blah b
        for ignored_label_key in self.labels_to_ignore_during_comparison:
            if ignored_label_key in label_keys:
                label_keys.remove(ignored_label_key)

            if ignored_label_key in other_label_keys:
                other_label_keys.remove(ignored_label_key)

        if len(set(label_keys)) != len(set(other_label_keys)):
            return False

        for label_key in label_keys:
            if label_key in self.labels_to_ignore_during_comparison:
                continue

            if label_key not in other_label_and_boxes:
                return False

            matches = []

            rects = label_and_boxes[label_key]
            other_rects = other_label_and_boxes[label_key]

            if len(rects) != len(other_rects):
                return False

            overlap_threshold = rect_overlap_percent

            for rect in rects:
                match_found = False
                for other_rect in other_rects:
                    if calculate_overlap(rect, other_rect) > overlap_threshold:
                        match_found = True
                        break

                if match_found == False:
                    return False

        return True


    def get_colors(self, LABELS):
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
        return COLORS

    def store_frame(self, original_filename, frame, label_and_boxes, assisted_labels, timestamp=None, video_filename=None):
        self.create_session_folders()

        is_first_of_kind = False
        is_second_of_kind = False
        inside_of_first_of_a_kind = False
        for other_label_and_boxes in self.first_of_kinds:
            inside_of_first_of_a_kind = inside_of_first_of_a_kind or self.is_same(label_and_boxes, other_label_and_boxes)
            if inside_of_first_of_a_kind:
                break

        inside_of_second_of_a_kind = False
        for other_label_and_boxes in self.second_of_kinds:
            inside_of_second_of_a_kind = inside_of_second_of_a_kind or self.is_same(label_and_boxes, other_label_and_boxes)
            if inside_of_second_of_a_kind:
                break

        if not inside_of_first_of_a_kind and not inside_of_second_of_a_kind:
            self.first_of_kinds.append(label_and_boxes)
            is_first_of_kind = True

        elif not inside_of_second_of_a_kind:
            self.second_of_kinds.append(label_and_boxes)
            is_second_of_kind = True

        labels_and_boxes = []
        drawn_image = frame.copy()

        yolo_formats = []

        num_of_rects = 0
        # ensure at least one detection exists
        for label_name in label_and_boxes:
            rects = label_and_boxes[label_name]
            # loop over the indexes we are keeping
            for rect in rects:
                num_of_rects += 1
                # extract the bounding box coordinates
                (x1, y1) = (rect[0], rect[1])
                (x2, y2) = (rect[2], rect[3])

                color_index = self.all_label_names.index(label_name)

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.colors[color_index]]
                cv2.rectangle(drawn_image, (x1, y1), (x2, y2), color, 2)
                text = "{}: {:.4f}".format(label_name, 0)
                cv2.putText(drawn_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)


            centroid, width, height = get_center_point_width_height(rect)
            xCenter, yCenter = centroid

            output = '%s %.6f %.6f %.6f %.6f\n' % (label_name, xCenter, yCenter, width, height)

            print(output)
            yolo_formats.append(output)

        filename = original_filename
        if len(assisted_labels) > 0:
            filename = 'assisted_%s' % original_filename

        dir_to_use = self.labeled_images_session_dir
        if not os.path.exists(os.path.dirname(dir_to_use)):
            os.makedirs(os.path.dirname(dir_to_use))

        write_filename = '%s/%s' % (dir_to_use, filename)

        status = cv2.imwrite(write_filename, drawn_image)

        dir_to_use = self.images_session_dir
        if not os.path.exists(os.path.dirname(dir_to_use)):
            os.makedirs(os.path.dirname(dir_to_use))
        write_filename = '%s/%s' % (dir_to_use, filename)

        status = cv2.imwrite(write_filename, frame)

        dir_to_use = self.yolo_label_session_dir
        filename_noext = filename.split('.')[0]
        textfile_out = '%s/%s.txt' % (dir_to_use, filename_noext)

        f = open(textfile_out, "w")
        f.writelines(yolo_formats)

        label_keys = label_and_boxes.keys()

        # check last frame for dramatic shift detection.
        dramatic_shift = False
        if self.last_label_keys is not None:
            if len(set(label_keys)) != len(set(self.last_label_keys)):
                dramatic_shift = True
        self.last_label_keys = label_keys


        csv_data = {
            'filename': original_filename,
            'assisted': len(assisted_labels) > 0,
            'assisted_labels': assisted_labels,
            'seen_labels': list(set(label_keys)),
            'num_of_rects': num_of_rects,
            'first_of_a_kind': is_first_of_kind,
            'second_of_a_kind': is_second_of_kind,
            "dramatic_shift": dramatic_shift,
        }
        if timestamp is not None:
            csv_data['timestamp'] = timestamp

        self.json_generator.append(csv_data)

        frame_str = str(original_filename)

        is_assisted = len(assisted_labels) > 0
        if timestamp is not None:
            if filename not in self.video_assets:
                self.video_assets[filename] = []
            self.video_assets[filename].append(
                    {
                        "filename": "",
                        "labels_and_boxes_key": frame_str,
                        "required_testing": is_second_of_kind,
                        "required_training": is_first_of_kind,
                        "is_assisted": is_assisted,
                        "dramatic_shift": dramatic_shift,
                        "timestamp": timestamp,
                    })
        else:
            self.image_asset_data.append({
                "filename": original_filename,
                "labels_and_boxes_key": frame_str,
                "required_testing": is_second_of_kind,
                "required_training": is_first_of_kind,
                "is_assisted": is_assisted,
                "dramatic_shift": dramatic_shift,
            })

        height, width, _ = frame.shape
        self.frames_and_label_and_boxes[frame_str] = {}
        self.frames_and_label_and_boxes[frame_str]['assisted'] = is_assisted
        self.frames_and_label_and_boxes[frame_str]['labels'] = label_and_boxes
        self.frames_and_label_and_boxes[frame_str]['artifacts_filename'] = filename
        self.frames_and_label_and_boxes[frame_str]['manual'] = False
        self.frames_and_label_and_boxes[frame_str]['width'] = width
        self.frames_and_label_and_boxes[frame_str]['height'] = height
        self.frames_and_label_and_boxes[frame_str]['required_testing'] = is_second_of_kind
        self.frames_and_label_and_boxes[frame_str]['required_training'] = is_first_of_kind

        self.labels_and_boxes_key_mapping[frame_str] = label_and_boxes

    def wrapup(self):
        print('writing metrics')
        self.metrics_session_dir = '%s/%s' % (METRICS_DIR, self.session_id)

        with open('%s/%s' % (self.metrics_session_dir, 'metrics.json'), "w") as outfile:
            json.dump(self.json_generator, outfile, indent=4)

        vott_csv_filename = '%s_vott_export.csv' % (self.session_id)
        vott_csv_filename = 'vott_export.csv'

        with open('%s/%s' % (self.metrics_session_dir, vott_csv_filename), "w") as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow(["image","xmin","ymin","xmax","ymax","label", "width", "height"])

            for filename_key in self.frames_and_label_and_boxes:
                output_data = self.frames_and_label_and_boxes[filename_key]

                for output_label in output_data['labels']:
                    for output_rect in output_data['labels'][output_label]:
                        imagename = output_data['artifacts_filename']
                        xmin = output_rect[0]
                        ymin = output_rect[1]
                        xmax = output_rect[2]
                        ymax = output_rect[3]
                        label = output_label
                        width = output_data['width']
                        height = output_data['height']

                        csvwriter.writerow([imagename, xmin, ymin, xmax, ymax, label, width, height])

        required_testing_filename = '-testing-export.txt'
        required_testing_filename = '%s_testing-export.txt' % (self.session_id)
        with open('%s/%s' % (self.metrics_session_dir, required_testing_filename), "w") as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow(["name"])

            for filename_key in self.frames_and_label_and_boxes:
                output_data = self.frames_and_label_and_boxes[filename_key]

                if output_data['required_testing']:
                    csvwriter.writerow([output_data['artifacts_filename']])


        required_training_filename = '-training-export.txt'
        required_training_filename = '%s_training-export.txt' % (self.session_id)
        with open('%s/%s' % (self.metrics_session_dir, required_training_filename), "w") as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow(["name"])

            for filename_key in self.frames_and_label_and_boxes:
                output_data = self.frames_and_label_and_boxes[filename_key]

                if output_data['required_training']:
                    csvwriter.writerow([output_data['artifacts_filename']])




    # ignore processed/rejected text should always be saved elsewhere
    def update_state(self, frame_key, label_and_boxes):
        self.labels_and_boxes_key_mapping[frame_key] = label_and_boxes
        self.frames_and_label_and_boxes[frame_key]['manual'] = True
        self.frames_and_label_and_boxes[frame_key]['labels'] = label_and_boxes


def get_center_point_width_height(rect):
	width = rect[2] - rect[0]
	height = rect[3] - rect[1]
	startX, startY, endX, endY = rect
	cX = int((startX + endX) / 2.0)
	cY = int((startY + endY) / 2.0)

	return [cX, cY], width, height

def distance_between_points(p1, p2):
	return math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )

def calculate_overlap(rect_1, rect_2):
    XA2 = rect_1[2]
    XB2 = rect_2[2]
    XA1 = rect_1[0]
    XB1 = rect_2[0]

    YA2 = rect_1[3]
    YB2 = rect_2[3]
    YA1 = rect_1[1]
    YB1 = rect_2[1]
    # overlap between A and B
    SA = (rect_1[2] - rect_1[0]) * (rect_1[3] - rect_1[1])
    SB = (rect_2[2] - rect_2[0]) * (rect_2[3] - rect_2[1])

    SI = max(0, 1+ min(XA2, XB2) - max(XA1, XB1)) * max(0, 1 + min(YA2, YB2) - max(YA1, YB1))
    SU = SA + SB - SI
    if float(SA) == 0.0:
        return 0
    return SI/float(SA)
