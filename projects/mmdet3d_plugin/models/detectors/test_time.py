from mmdet.models import DETECTORS, build_backbone
from mmdet.models.detectors.fcos import FCOS
from mmdet.core import bbox2result
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.detectors import BaseDetector

@DETECTORS.register_module(force=True)
class TestTime(BaseDetector):
    def __init__(self, backbone, train_cfg=None, test_cfg=None, **kwargs):
        super(TestTime, self).__init__(**kwargs)
        self.backbone = build_backbone(backbone)

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def simple_test(self, img, img_metas, rescale=False):
        feat = self.extract_feat(img)
        return feat

    def aug_test(self):
        pass

    # def simple_test(self, img, img_metas, rescale=False):
    #     """Test function without test-time augmentation.

    #     Args:
    #         img (torch.Tensor): Images with shape (N, C, H, W).
    #         img_metas (list[dict]): List of image information.
    #         rescale (bool, optional): Whether to rescale the results.
    #             Defaults to False.

    #     Returns:
    #         list[list[np.ndarray]]: BBox results of each image and classes.
    #             The outer list corresponds to each image. The inner list
    #             corresponds to each class.
    #     """
    #     feat, _ = self.extract_feat(img)
    #     results_list = self.bbox_head.simple_test(
    #         feat, img_metas, rescale=rescale)
    #     bbox_results = [
    #         (bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes), metas['image_id'][0])
    #         for (det_bboxes, det_labels), metas in zip(results_list, img_metas)
    #     ]
    #     # bbox_pts = [
    #     #     (self.bbox2result2d(det_bboxes, det_labels, self.img_roi_head.num_classes), image_id)
    #     #     for (det_bboxes, det_labels), image_id in bbox_pts
    #     #     ]
    #     return bbox_results

    def bbox2result2d(self, bboxes, labels, num_classes):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor | np.ndarray): shape (n, 5)
            labels (torch.Tensor | np.ndarray): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class
        """
        if bboxes.shape[0] == 0:
            return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
        else:
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
            return [bboxes[labels == i, :] for i in range(num_classes)]

    def show_result(self, data, result, **kwargs):
        """
        data,
        result,
        out_dir=out_dir,
        show=show,
        score_thr=show_score_thr
        """
        import cv2
        import os
        import numpy as np
        import random
        token = result[1]
        bbox = result[0]
        img = data
        for label, boxes in enumerate(bbox):
            for box in boxes:
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
        
        cv2.imwrite(f'/root/tmp/vis_2d/{token}.jpg',img)