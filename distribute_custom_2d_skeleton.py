# Copyright (c) OpenMMLab. All rights reserved.
# coding=utf-8
# coding=gbk
import argparse
import os
import os.path as osp
import pyskl
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_top_down_pose_model, init_pose_model
import decord
import mmcv
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import cv2
from pyskl.smp import mrlines


def extract_frame(video_path):
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]


def detection_inference(model, frames):
    model = model.cuda()
    results = []
    for frame in frames:
        result = inference_detector(model, frame)
        results.append(result)
    return results


def pose_inference(model, frames, det_results):
    model = model.cuda()
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = max([len(x) for x in det_results])
    kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32)

    for i, (f, d) in enumerate(zip(frames, det_results)):
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        for j, item in enumerate(pose):
            kp[j, i] = item['keypoints']
    return kp


pyskl_root = osp.dirname(pyskl.__path__[0])
default_det_config = f'{pyskl_root}/demo/faster_rcnn_r50_fpn_1x_coco-person.py'
default_det_ckpt = (
    'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
    'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth')
default_pose_config = f'{pyskl_root}/demo/td-hm_litehrnet-18_8xb64-210e_coco-256x192.py'
default_pose_ckpt = (
    '/root/pyskl/demo/litehrnet18_coco_256x192-6bace359_20211230.pth')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    parser.add_argument(
        '--det-config',
        default=default_det_config,
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-ckpt',
        default=default_det_ckpt,
        help='human detection checkpoint file/url')
    parser.add_argument('--pose-config', type=str, default=default_pose_config)
    parser.add_argument('--pose-ckpt', type=str, default=default_pose_ckpt)
    parser.add_argument('--det-score-thr', type=float, default=0.7)
    parser.add_argument('--det-area-thr', type=float, default=1300)
    parser.add_argument('--video-list', default=r'/root/pyskl/until/no_miss.list', type=str, help='the list of source videos')
    parser.add_argument('--out', default=r'/root/pyskl/until/no_miss_ntu_train.pkl', type=str, help='output pickle name')
    parser.add_argument('--tmpdir', type=str, default='tmp')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num-gpus', type=int, default=1, help='number of GPUs to use')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out.endswith('.pkl')

    dist.init_process_group(backend='nccl', world_size=args.num_gpus)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(args.local_rank)

    lines = mrlines(args.video_list)
    lines = [x.split() for x in lines]

    assert len(lines[0]) in [1, 2]
    if len(lines[0]) == 1:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0]) for x in lines]
    else:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0], label=int(x[1])) for x in lines]

    my_part = annos[rank::world_size]

    if rank == 0:
        os.makedirs(args.tmpdir, exist_ok=True)
    dist.barrier()

    det_model = init_detector(args.det_config, args.det_ckpt, 'cuda')
    assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
    pose_model = init_pose_model(args.pose_config, args.pose_ckpt, 'cuda')

    for anno in tqdm(my_part):
        frames = extract_frame(anno['filename'])
        det_results = detection_inference(det_model, frames)
        det_results = [x[0] for x in det_results]
        for i, res in enumerate(det_results):
            res = res[res[:, 4] >= args.det_score_thr]
            box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
            assert np.all(box_areas >= 0)
            res = res[box_areas >= args.det_area_thr]
            det_results[i] = res

        pose_results = pose_inference(pose_model, frames, det_results)
        shape = frames[0].shape[:2]
        anno['img_shape'] = anno['original_shape'] = shape
        anno['total_frames'] = len(frames)
        anno['num_person_raw'] = pose_results.shape[0]
        anno['keypoint'] = pose_results[..., :2].astype(np.float16)
        anno['keypoint_score'] = pose_results[..., 2].astype(np.float16)
        anno.pop('filename')

    mmcv.dump(my_part, osp.join(args.tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    if rank == 0:
        parts = [mmcv.load(osp.join(args.tmpdir, f'part_{i}.pkl')) for i in range(world_size)]
        rem = len(annos) % world_size
        if rem:
            for i in range(rem, world_size):
                parts[i].append(None)

        ordered_results = []
        for res in zip(*parts):
            ordered_results.extend(list(res))
        ordered_results = ordered_results[:len(annos)]
        mmcv.dump(ordered_results, args.out)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
