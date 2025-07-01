from typing import Optional, List, Union

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.ops.boxes import box_area
from torchvision.transforms import functional as F

import random
from einops import rearrange, repeat
from PIL import Image, ImageFile
import numpy as np
from icecream import ic

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


def box_iou(boxes1, area1, boxes2, eps=1e-5):
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + eps)
    return iou, union


def anchor_rank(anchors, anchors_areas, input_image_size, eps=1e-5):
    # anchors x1 y1 x2 y2

    # image_size: (h, w)
    # xyxy
    input_image_bbox = torch.tensor([0, 0, input_image_size[1], input_image_size[0]]).unsqueeze(0)

    boxes1 = anchors
    boxes2 = input_image_bbox
    boxes3 = anchors.clone()
    # y2
    boxes3[:, 3] = input_image_size[0] / input_image_size[1] * anchors[:, 2]  # 用于算分辨率无关的iou

    area1 = anchors_areas

    iou, _ = box_iou(boxes1, area1, boxes2)
    iou = iou.squeeze(1)
    shape_iou, _ = box_iou(boxes1, area1, boxes3)
    shape_iou = shape_iou.diag()
    # 优先匹配形状接近 再匹配分辨率接近
    index = torch.argmax(shape_iou * 100 + iou, dim=0)
    return index


class AnchorResize(torch.nn.Module):

    def __init__(self, image_size, anchors, interpolation=InterpolationMode.BILINEAR, antialias=None):
        super().__init__()
        # xyxy
        self.anchors = torch.tensor(
            [[0, 0, _[1] * image_size[1], _[0] * image_size[0]]
             for _ in anchors], requires_grad=False
        )

        self.anchor_areas = box_area(self.anchors)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img, skip_resize=False):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        selected_anchor = anchor_rank(self.anchors, self.anchor_areas, (img.size[1], img.size[0]))
        target_size = self.anchors[selected_anchor][2:].tolist()  # w,h
        if skip_resize:
            # for debug
            return selected_anchor
        return F.resize(img, [target_size[1], target_size[0]], self.interpolation, max_size=None,
                        antialias=self.antialias), selected_anchor

    def __repr__(self) -> str:
        detail = f"(size={self.image_size}, anchor={self.anchors}, interpolation={self.interpolation.value}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


# grid_dict = {
#     'grid_1': [
#         (1, 1)],
#     'grid_4': [
#         (1, 1),
#         (1, 2), (2, 1),
#         (1, 3), (3, 1),
#         (2, 2), (1, 4), (4, 1)],
#     'grid_9': [
#         (1, 1),
#         (1, 2), (2, 1),
#         (1, 3), (3, 1),
#         (2, 2), (1, 4), (4, 1),
#         (1, 5), (5, 1),
#         (1, 6), (6, 1), (2, 3), (3, 2),
#         (1, 7), (7, 1),
#         (4, 2), (2, 4), (1, 8), (8, 1),
#         (3, 3), (1, 9), (9, 1)],
#     'grid_3x3': [
#         (3, 3)],
#     'grid_20': [
#         (1, 1),
#         (1, 2), (2, 1),
#         (1, 3), (3, 1), (1, 4), (2, 2), (4, 1),
#         (1, 5), (5, 1),
#         (1, 6), (2, 3), (3, 2), (6, 1),
#         (1, 7), (7, 1),
#         (1, 8), (2, 4), (4, 2), (8, 1),
#         (1, 9), (3, 3), (9, 1),
#         (1, 10), (2, 5), (5, 2), (10, 1),
#         (1, 11), (11, 1),
#         (2, 6), (3, 4), (4, 3), (6, 2),
#         (2, 7), (7, 2),
#         (3, 5), (5, 3),
#         (2, 8), (4, 4), (8, 2),
#         (2, 9), (3, 6), (6, 3), (9, 2),
#         (2, 10), (4, 5), (5, 4), (10, 2)]
# }

N_grid = 28
grids = [
    [(i, n // i) for i in range(1, n + 1) if n % i == 0]
    for n in range(1, N_grid)
]
grid_dict = {
    f'grid_{n}': sum(grids[:n], [])
    for n in range(1, N_grid)
}


""" borrowed from mplug-docowl1.5
"""
class AdaptiveCropProcessor():
    def __init__(
        self,
        image_size=224,
        anchors='grid_9',
        add_global_img=True,
        add_textual_crop_indicator=False,
        enable_low_res=False
    ):
        self.add_global_img = add_global_img
        self.add_textual_crop_indicator = add_textual_crop_indicator
        self.media_token = "<image>"

        self.crop_size = {'width': image_size, 'height': image_size}  # compatible with old codes

        # h,w
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        # h,w
        anchors = grid_dict[anchors]
        self.anchors = [tuple(_) for _ in anchors]
        self.anchor_max = max([max(_) for _ in self.anchors])
        # xywh -> xyxy
        self.resizer = AnchorResize(image_size=image_size, anchors=anchors, interpolation=InterpolationMode.BICUBIC)
        self.old_resizer = transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.enable_low_res = enable_low_res

    def get_sub_images(self, image):
        image, selected_anchor = self.resizer(image)

        org_w, org_h = image.size
        tar_w, tar_h = self.image_size[1], self.image_size[0]
        num_w, num_h = org_w // tar_w, org_h // tar_h

        sub_images = []
        for i in range(num_h):
            for j in range(num_w):
                x1, y1 = j * tar_w, i * tar_h
                x2, y2 = x1 + tar_w, y1 + tar_h

                sub_img = image.crop((x1, y1, x2, y2))
                sub_images.append(sub_img)

        return sub_images

    def _process_image(self, images):
        new_images = []
        new_patch_position = []
        num_image_mult = []
        for image in images:
            if self.add_global_img:
                nocut_image = self.image_transform(self.old_resizer(image)).unsqueeze(0)

            image, selected_anchor = self.resizer(image)
            image_input = self.image_transform(image)  # h,w,3 -> 3,h,w
            # rearrange(x,'B C (n1 h) (n2 w) -> (B n1 n2) C h w', n1=self.down_sample[0], n2=self.down_sample[1])
            image_input = rearrange(image_input, 'C (num_h h) (num_w w) -> (num_h num_w) C h w', h=self.image_size[0],
                                    w=self.image_size[1])

            if self.add_global_img:
                image_input = torch.cat([nocut_image, image_input], dim=0)

            anchor = self.anchors[selected_anchor]  # w,h
            patch_position = torch.cat([
                repeat(torch.arange(anchor[0]), 'num_h -> num_h num_w 1', num_w=anchor[1]),
                repeat(torch.arange(anchor[1]), 'num_w -> num_h num_w 1', num_h=anchor[0])], dim=2)
            patch_position = rearrange(patch_position, 'num_h num_w p-> (num_h num_w) p', p=2)  # num_patch, (ph,pw)

            if self.add_global_img:
                patch_position = torch.cat([torch.ones(1, 2).long() * self.anchor_max, patch_position], dim=0)

            new_images.append(image_input)
            new_patch_position.append(patch_position)
            num_image_mult.append(patch_position.shape[0])

        new_images = torch.cat(new_images, dim=0)
        new_patch_position = torch.cat(new_patch_position, dim=0)
        return new_images, new_patch_position, num_image_mult

    def __call__(self, images=None, query=None):
        assert images is not None

        # if isinstance(images, str):
        #     images = [images]
        # image_pils = []
        # for image_url in images:
        #     image = Image.open(image_url).convert('RGB')
        #     # ic(image.size)
        #     image_pils.append(image)

        if not isinstance(images, list):
            images = [images]

        image_pils = []
        if isinstance(images[0], str):
            for image_url in images:
                image = Image.open(image_url).convert('RGB')
                # ic(image.size)
                image_pils.append(image)
        elif isinstance(images[0], np.ndarray):
            if len(images[0].shape) == 4:  # [(N, H, W, C)]
                images = images[0]
            image_pils = [Image.fromarray(img) for img in images]
        else:
            image_pils = images

        if self.enable_low_res:
            image_data = torch.cat([
                self.image_transform(self.old_resizer(image)).unsqueeze(0)
                for image in image_pils
            ], dim=0)
            assert image_data.ndim == 4

            patch_position = torch.ones((image_data.shape[0], 2), dtype=torch.long) * self.anchor_max
            num_image_mult = [1 for _ in range(image_data.shape[0])]
        else:
            image_data, patch_position, num_image_mult = self._process_image(image_pils)

        assert self.media_token in query
        text_list = query.split(self.media_token)
        text = text_list[0]
        image_token_ptr = 0
        patch_pos_ptr = 0
        for next_text in text_list[1:]:
            n_img = num_image_mult[image_token_ptr]
            if self.add_textual_crop_indicator:
                # generate image placeholders with interleaved texutual crop indicator
                # e.g. <global_img><image><crop_img_row0_col0><image><crop_img_row0_col1><image>...
                for patch_pos in patch_position.tolist()[patch_pos_ptr: patch_pos_ptr+n_img]:
                    # global non-crop image
                    if patch_pos[0] == self.anchor_max and patch_pos[1] == self.anchor_max:
                        text += f'<global_img>{self.media_token}'
                    else:
                        row_col = 'row' + str(patch_pos[0]) + '_col' + str(patch_pos[1])
                        text += '<crop_img_' + row_col + f'>{self.media_token}'
            else:
                # generate successive image placeholders for a image, 1 crop img == 1 <image>
                text += self.media_token * n_img
            text += next_text
            image_token_ptr += 1
            patch_pos_ptr += n_img

        global_image = torch.cat([
            self.image_transform(self.old_resizer(image)).unsqueeze(0)
            for image in image_pils
        ], dim=0)
        assert global_image.ndim == 4

        data_dict = dict(
            global_image=global_image,
            cropped_images=image_data,
            patch_positions=patch_position,
            text=text
        )

        return data_dict


if __name__ == '__main__':
    for k, v in grid_dict.items():
        print("=========")
        print(k)
        print(v)