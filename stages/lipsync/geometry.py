"""Face crop geometry and spatial feathering from TaoLiveAIGC/TBDub.

Apache-2.0; see licenses/TBDub.txt. Extracted from preprocessing.py,
preprocessing_mediapipe.py and video_utils.py; inference lives in LightX2V.
Modified here to remove video decoding/model dependencies.
"""
import cv2
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter

HEIGHT = WIDTH = 512
VERTICAL_BBOX_SHIFT_RATIO = 0.0

def _normalize_window_length(window_length, size, minimum=3):
    if size <= 1:
        return 1
    window_length = min(window_length, size)
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(minimum, window_length)
    if window_length > size:
        window_length = size if size % 2 == 1 else size - 1
    return max(1, window_length)


def _to_float_scalar(value):
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(array[0])


def window_smooth(data_list, window_size=5):
    if len(data_list) <= 1:
        return data_list
    smoothed_data_list = [None] * len(data_list)
    for frame_id in range(len(data_list)):
        start = max(0, frame_id - window_size // 2)
        end = min(len(data_list), frame_id + window_size // 2 + 1)
        valid_data = [data_list[index] for index in range(start, end) if data_list[index] is not None]
        if len(valid_data) == 0:
            smoothed_data_list[frame_id] = None
        else:
            smoothed_data_list[frame_id] = np.mean(valid_data, axis=0).astype(np.float32)
    return smoothed_data_list


def sg_smooth(points, window_length=5, polyorder=2):
    if len(points) <= 2:
        return [np.array(point) for point in points]

    window_length = _normalize_window_length(window_length, len(points), minimum=3)
    polyorder = min(polyorder, window_length - 1)
    if window_length <= polyorder:
        return [np.array(point) for point in points]

    pad_len = window_length // 2
    points_array = np.array(points)
    points_len = len(points_array)
    points_array = points_array.reshape(points_len, -1)

    smoothed_dims = []
    for dim_id in range(points_array.shape[1]):
        dim_values = points_array[:, dim_id]
        dim_padded = np.pad(dim_values, (pad_len, pad_len), mode="edge")
        dim_smoothed_padded = savgol_filter(dim_padded, window_length, polyorder)
        dim_smoothed = dim_smoothed_padded[pad_len:-pad_len]
        smoothed_dims.append(dim_smoothed)

    smoothed_points = []
    for point_id in range(len(points)):
        point = [smoothed_dims[dim_id][point_id] for dim_id in range(points_array.shape[1])]
        smoothed_points.append(np.array(point, dtype=np.float32))
    return smoothed_points


def build_face_bbox_from_landmarks(face_ldmk, ori_width, ori_height, num_passes=5):
    num_frames, num_points, _ = face_ldmk.shape
    smoothed_ldmk = face_ldmk.copy()

    base_window = _normalize_window_length(15, num_frames, minimum=5)
    for pass_idx in range(num_passes):
        window_length = _normalize_window_length(base_window + pass_idx * 4, num_frames, minimum=5)
        polyorder = min(2, window_length - 1)
        if window_length <= polyorder:
            continue

        temp_ldmk = np.zeros_like(smoothed_ldmk)
        for point_id in range(num_points):
            temp_ldmk[:, point_id, 0] = savgol_filter(
                smoothed_ldmk[:, point_id, 0],
                window_length=window_length,
                polyorder=polyorder,
                mode="mirror",
            )
            temp_ldmk[:, point_id, 1] = savgol_filter(
                smoothed_ldmk[:, point_id, 1],
                window_length=window_length,
                polyorder=polyorder,
                mode="mirror",
            )
        smoothed_ldmk = temp_ldmk

    for window_size in (17, 7):
        window_size = min(window_size, num_frames)
        if window_size > 1:
            final_smoothed_ldmk = np.zeros_like(smoothed_ldmk)
            for frame_id in range(num_frames):
                start_idx = max(0, frame_id - window_size // 2)
                end_idx = min(num_frames, frame_id + window_size // 2 + 1)
                final_smoothed_ldmk[frame_id] = np.mean(smoothed_ldmk[start_idx:end_idx], axis=0)
            smoothed_ldmk = final_smoothed_ldmk

    # MediaPipe already includes the forehead in its face mesh.
    final_smoothed_ldmk = smoothed_ldmk

    bbox = []
    for frame_id in range(num_frames):
        frame_ldmk = final_smoothed_ldmk[frame_id]
        x_coords = np.clip(frame_ldmk[:, 0], 0, 1) * ori_width
        y_coords = np.clip(frame_ldmk[:, 1], 0, 1) * ori_height

        x1 = float(np.min(x_coords))
        y1 = float(np.min(y_coords))
        x2 = float(np.max(x_coords))
        y2 = float(np.max(y_coords))

        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        max_side = float(max(width, height) * 1.45) # 1.45

        new_x1 = center_x - max_side / 2
        new_y1 = center_y - max_side / 2
        new_x2 = center_x + max_side / 2
        new_y2 = center_y + max_side / 2

        if new_x1 < 0:
            new_x1 = 0
            new_x2 = min(max_side, ori_width)
        if new_x2 > ori_width:
            new_x2 = ori_width
            new_x1 = max(0, ori_width - max_side)
        if new_y1 < 0:
            new_y1 = 0
            new_y2 = min(max_side, ori_height)
        if new_y2 > ori_height:
            new_y2 = ori_height
            new_y1 = max(0, ori_height - max_side)

        if ori_width < max_side or ori_height < max_side:
            min_dim = float(min(ori_width, ori_height))
            new_x1 = center_x - min_dim / 2
            new_y1 = center_y - min_dim / 2
            new_x2 = new_x1 + min_dim
            new_y2 = new_y1 + min_dim

            if new_x1 < 0:
                new_x1 = 0
                new_x2 = min_dim
            if new_x2 > ori_width:
                new_x2 = ori_width
                new_x1 = ori_width - min_dim
            if new_y1 < 0:
                new_y1 = 0
                new_y2 = min_dim
            if new_y2 > ori_height:
                new_y2 = ori_height
                new_y1 = ori_height - min_dim

        bbox.append([float(new_x1), float(new_y1), float(new_x2), float(new_y2)])

    if len(bbox) > 1:
        bbox_smooth_window = min(7, len(bbox))
        smoothed_bbox = []
        for frame_id in range(len(bbox)):
            start_idx = max(0, frame_id - bbox_smooth_window // 2)
            end_idx = min(len(bbox), frame_id + bbox_smooth_window // 2 + 1)
            window_bboxes = np.array(bbox[start_idx:end_idx])
            avg_bbox = np.mean(window_bboxes, axis=0).astype(np.float32)
            width = avg_bbox[2] - avg_bbox[0]
            height = avg_bbox[3] - avg_bbox[1]
            if width != height:
                max_side = max(width, height)
                center_x = (avg_bbox[0] + avg_bbox[2]) / 2
                center_y = (avg_bbox[1] + avg_bbox[3]) / 2
                avg_bbox[0] = center_x - max_side / 2
                avg_bbox[1] = center_y - max_side / 2
                avg_bbox[2] = center_x + max_side / 2
                avg_bbox[3] = center_y + max_side / 2
            smoothed_bbox.append(avg_bbox.tolist())
        bbox = smoothed_bbox

    return bbox


def judge_case_flag(bbox_list):
    bboxes = np.array(bbox_list)
    threshold = 0.16
    max_bbox = np.array([np.min(bboxes[:, 0]), np.min(bboxes[:, 1]), np.max(bboxes[:, 2]), np.max(bboxes[:, 3])])
    width = max_bbox[2] - max_bbox[0]
    height = max_bbox[3] - max_bbox[1]
    tolerance_x = width * threshold
    tolerance_y = height * threshold
    case_1_flag = True
    for bbox in bboxes:
        diff_x1 = abs(bbox[0] - max_bbox[0])
        diff_x2 = abs(max_bbox[2] - bbox[2])
        diff_y1 = abs(bbox[1] - max_bbox[1])
        diff_y2 = abs(max_bbox[3] - bbox[3])
        if diff_x1 > tolerance_x or diff_x2 > tolerance_x or diff_y1 > tolerance_y or diff_y2 > tolerance_y:
            case_1_flag = False
            break
    if case_1_flag:
        return 1

    sides = bboxes[:, 2] - bboxes[:, 0]
    avg_side = np.mean(sides)
    threshold = 0.1
    lower_bound = avg_side * (1 - threshold)
    upper_bound = avg_side * (1 + threshold)
    case_2_flag = True
    for side in sides:
        if side < lower_bound or side > upper_bound:
            case_2_flag = False
            break
    if case_2_flag:
        return 2

    return 3


def get_fix_bbox(bbox_list, ori_width, ori_height):
    bbox_list = window_smooth(bbox_list, window_size=5)
    bboxes = np.array(bbox_list)
    x1 = np.min(bboxes[:, 0])
    y1 = np.min(bboxes[:, 1])
    x2 = np.max(bboxes[:, 2])
    y2 = np.max(bboxes[:, 3])

    side = max(x2 - x1, y2 - y1) * 1.0 #1.0
    side = min(side, ori_width, ori_height)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    x1 = center_x - side / 2
    x2 = center_x + side / 2
    y1 = center_y - side / 2
    y2 = center_y + side / 2

    if x1 < 0:
        x1 = 0
        x2 = x1 + side
    if x2 > ori_width:
        x2 = ori_width
        x1 = x2 - side
    if y1 < 0:
        y1 = 0
        y2 = y1 + side
    if y2 > ori_height:
        y2 = ori_height
        y1 = y2 - side

    return [[float(x1), float(y1), float(x2), float(y2)] for _ in range(len(bboxes))]


def get_size_fix_center_smooth_bbox(bbox_list, height, width, sg_win, sg_order, avg_win):
    bbox_list = window_smooth(bbox_list, window_size=5)
    bboxes = np.array(bbox_list)
    bbox_center_list = []
    bbox_side_list = []
    for bbox in bboxes:
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        side = bbox[2] - bbox[0]
        bbox_center_list.append(np.array([center_x, center_y]))
        bbox_side_list.append(side)

    bbox_center_list = window_smooth(bbox_center_list, avg_win+4) # 15 
    bbox_center_list = sg_smooth(bbox_center_list, sg_win, sg_order)   # 25 2
    bbox_center_list = window_smooth(bbox_center_list, avg_win)  # 11
    bbox_center_list = window_smooth(bbox_center_list, avg_win-4)   # 7
    bbox_center_list = window_smooth(bbox_center_list, avg_win-4-4)   # 3

    max_side = max(bbox_side_list)
    max_side = min(max_side, height, width)
    max_side = float(max_side)

    smoothed_bbox_list = []
    for center in bbox_center_list:
        cx = center[0]
        cy = center[1]
        side = max_side
        x1 = cx - side / 2
        x2 = cx + side / 2
        y1 = cy - side / 2
        y2 = cy + side / 2
        if x1 < 0:
            x1 = 0
            x2 = x1 + side
        if x2 > width:
            x2 = width
            x1 = x2 - side
        if y1 < 0:
            y1 = 0
            y2 = y1 + side
        if y2 > height:
            y2 = height
            y1 = y2 - side
        smoothed_bbox_list.append([float(x1), float(y1), float(x2), float(y2)])
    return smoothed_bbox_list


def get_smooth_bbox(bbox_list, height, width, sg_win, sg_order, avg_win):
    bbox_list = window_smooth(bbox_list, window_size=5)  # 5
    bbox_list = window_smooth(bbox_list, avg_win)   # 11
    bbox_list = sg_smooth(bbox_list, sg_win, sg_order)
    bbox_list = window_smooth(bbox_list, avg_win-4)  # 7
    bbox_list = window_smooth(bbox_list, avg_win-4-4)  # 3  

    side_list = []
    for bbox in bbox_list:
        side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
        side_list.append(side)
    side_list = window_smooth(side_list, avg_win)
    side_list = sg_smooth(side_list, sg_win, sg_order)
    side_list = window_smooth(side_list, avg_win - 4)

    smoothed_bbox_list = []
    for bbox, side in zip(bbox_list, side_list):
        x1, y1, _, _ = bbox
        x1 = _to_float_scalar(x1)
        y1 = _to_float_scalar(y1)
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        side = _to_float_scalar(side)
        side = min(side, height - y1, width - x1)
        x2 = x1 + side
        y2 = y1 + side
        smoothed_bbox_list.append(np.array([x1, y1, x2, y2], dtype=np.float32))

    smoothed_bbox_list = window_smooth(smoothed_bbox_list, window_size=avg_win)  # 11
    smoothed_bbox_list = window_smooth(smoothed_bbox_list, window_size=avg_win-4)  # 7
    smoothed_bbox_list = window_smooth(smoothed_bbox_list, window_size=avg_win-4-4)  # 7

    bbox_list = []
    for bbox in smoothed_bbox_list:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        bbox_list.append([float(x1), float(y1), float(x2), float(y2)])
    return bbox_list


def process_bbox(bbox_list, ori_width, ori_height, force_fix=False):
    if force_fix:
        case_flag = 0
    else:
        case_flag = judge_case_flag(bbox_list)

    if case_flag in (0, 1):
        bbox_list = get_fix_bbox(bbox_list, ori_width, ori_height)
    if case_flag == 2:
        bbox_list = get_size_fix_center_smooth_bbox(bbox_list, ori_height, ori_width, sg_win=25, sg_order=2, avg_win=11)
    if case_flag == 3:
        bbox_list = get_smooth_bbox(bbox_list, ori_height, ori_width, sg_win=25, sg_order=2, avg_win=11)

    finalized_bbox_list = [
        finalize_bbox_for_crop(
            apply_vertical_bbox_shift(bbox, ori_height, VERTICAL_BBOX_SHIFT_RATIO),
            ori_width,
            ori_height,
        )
        for bbox in bbox_list
    ]
    return finalized_bbox_list, case_flag


def finalize_bbox_for_crop(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(np.floor(x1))))
    y1 = max(0, min(height - 1, int(np.floor(y1))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x2))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y2))))
    return [x1, y1, x2, y2]


def apply_vertical_bbox_shift(bbox, image_height, shift_ratio):
    x1, y1, x2, y2 = [float(value) for value in bbox]
    shift = (y2 - y1) * shift_ratio
    y1 += shift
    y2 += shift

    if y2 > image_height:
        overflow = y2 - image_height
        y1 -= overflow
        y2 -= overflow
    if y1 < 0:
        overflow = -y1
        y1 += overflow
        y2 += overflow

    return [x1, y1, x2, y2]


def crop_and_resize_frames(raw_video, bbox_list):
    ref_video = []
    for frame, bbox in zip(raw_video, bbox_list):
        x1, y1, x2, y2 = bbox
        crop = frame.crop((x1, y1, x2, y2))
        ref_video.append(crop.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR))
    return ref_video


def fill_short_gaps(landmarks, max_gap_frames):
    """Interpolate short gaps; reject long or entirely missing face tracks."""
    valid = np.array([points is not None for points in landmarks], dtype=bool)
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise RuntimeError("MediaPipe detected no face in the video.")
    missing = np.flatnonzero(~valid)
    if len(missing):
        groups = np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1)
        if max(map(len, groups)) > max_gap_frames:
            raise RuntimeError(f"MediaPipe face track has a gap longer than {max_gap_frames} frames.")
    points = np.stack([landmarks[index] for index in indices])
    result = np.empty((len(landmarks), *points.shape[1:]), dtype=np.float32)
    for point in range(points.shape[1]):
        for coordinate in range(points.shape[2]):
            result[:, point, coordinate] = np.interp(
                np.arange(len(landmarks)), indices, points[:, point, coordinate]
            )
    return result, missing.tolist()


def crop_detected_face(pixels, box):
    """Return a 256px RGB crop and its pixel-to-original affine transform."""
    side = max(box.width, box.height) * 1.5
    if side <= 0:
        raise ValueError("MediaPipe returned an empty detection box.")
    x = box.origin_x + box.width / 2 - side / 2
    y = box.origin_y + box.height / 2 - side / 2
    matrix = np.array([[side / 256, 0, x], [0, side / 256, y]], dtype=np.float32)
    crop = cv2.warpAffine(
        pixels, matrix, (256, 256), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, matrix


def normalize_blend_edge(edge, height, width):
    max_edge = max(1, min(height, width) // 2 - 1)
    edge = max(1, min(edge, max_edge))
    if edge % 2 == 0:
        edge = edge + 1 if edge < max_edge else edge - 1
    return max(1, edge)


def build_blurred_border_mask(height, width, edge_up, edge_down, edge_left, edge_right, blur_kernel_size):
    blur_kernel_size = normalize_blend_edge(blur_kernel_size, height, width)
    mask = np.ones([height, width], dtype=np.float32)
    mask[edge_up:height - edge_down, edge_left:width - edge_right] = 0
    mask = cv2.blur(mask, (blur_kernel_size, blur_kernel_size))
    mask = cv2.blur(mask, (blur_kernel_size, blur_kernel_size))
    return mask[:, :, np.newaxis]


def apply_edge_mask(
    img_generated,
    img_reference,
    edge_margin,
    blur_kernel_size,
    is_up_all=False,
    is_below_all=False,
    is_left_all=False,
    is_right_all=False,
):
    height, width = img_generated.shape[:2]
    edge_margin = max(0, min(edge_margin, min(height, width) // 2 - 1))
    edge_up = 0 if is_up_all else edge_margin
    edge_down = 0 if is_below_all else edge_margin
    edge_left = 0 if is_left_all else edge_margin
    edge_right = 0 if is_right_all else edge_margin

    mask = build_blurred_border_mask(
        height,
        width,
        edge_up=edge_up,
        edge_down=edge_down,
        edge_left=edge_left,
        edge_right=edge_right,
        blur_kernel_size=blur_kernel_size,
    )
    mask = np.concatenate((mask, mask, mask), axis=2)

    blended = img_generated * (1 - mask) + img_reference * mask
    blended[blended > 255] = 255
    blended[blended < 0] = 0
    return blended.astype(np.uint8)


def apply_inner_edge_mask(img_generated, img_reference, pixel_margin):
    height, width = img_generated.shape[:2]
    mask = build_blurred_border_mask(
        height,
        width,
        edge_up=pixel_margin,
        edge_down=pixel_margin,
        edge_left=pixel_margin,
        edge_right=pixel_margin,
        blur_kernel_size=33,
    )
    mask = np.concatenate((mask, mask, mask), axis=2)

    blended = img_generated * (1 - mask) + img_reference * mask
    blended[blended > 255] = 255
    blended[blended < 0] = 0
    return blended.astype(np.uint8)


def blend_crop_video_with_ref(video, ref_video_frames, replace_border_latents_width):
    pixel_margin = replace_border_latents_width * 16
    num_frames = min(len(video), len(ref_video_frames))
    blended_frames = []
    for frame_idx in range(num_frames):
        generated_frame = np.array(video[frame_idx].convert("RGB"), dtype=np.float32)
        ref_frame = np.array(ref_video_frames[frame_idx].convert("RGB"), dtype=np.float32)
        blended_frame = apply_inner_edge_mask(generated_frame, ref_frame, pixel_margin)
        blended_frames.append(Image.fromarray(blended_frame))
    return blended_frames


def paste_video_back(video, raw_video_frames, bboxes):
    num_frames = min(len(video), len(raw_video_frames), len(bboxes))
    result_frames = []
    for frame_idx in range(num_frames):
        generated_frame = np.array(video[frame_idx].convert("RGB"), dtype=np.uint8)
        raw_frame = np.array(raw_video_frames[frame_idx].convert("RGB"), dtype=np.uint8).copy()
        x1, y1, x2, y2 = bboxes[frame_idx]
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        resized_generated = cv2.resize(generated_frame, (bbox_w, bbox_h))
        original_region = raw_frame[y1:y2, x1:x2].copy()
        blended_region = apply_edge_mask(
            resized_generated.astype(np.float32),
            original_region.astype(np.float32),
            edge_margin=8,
            blur_kernel_size=17,
            is_up_all=(y1 == 0),
            is_below_all=(y2 >= raw_frame.shape[0]),
            is_left_all=(x1 == 0),
            is_right_all=(x2 >= raw_frame.shape[1]),
        )
        raw_frame[y1:y2, x1:x2] = blended_region
        result_frames.append(Image.fromarray(raw_frame))
    return result_frames
