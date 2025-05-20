# LKAM-RCatenaryDet

### Foreign object detection model for railway catenary.

## If you use this code or dataset in your research, please cite the following paper:

## "LKAM-RCatenaryDet: Enhanced Railway Catenary Foreign Object Detection via Large Kernel Fusion and Global Attention Mechanism",  
## *The Journal of Supercomputing, 2025.
* The BibTex:
```bibtex
@article{ref,
  title     = {LKAM-RCatenaryDet: Enhanced Railway Catenary Foreign Object Detection via Large Kernel Fusion and Global Attention Mechanism},
  author    = {Wang, Yanjuan and Ji, Xianxin and Li, Jiatong and Zhao, Jun and Hu, Yuxin and Xu, Fengqiang and Li, Fengqi},
  journal   = {The Visual Computer},
  year      = {2025},
  publisher = {Springer}
}
```
---

## 1. Environment Dependency (requirements.txt)

'requirements.txt' has been provided in the root directory of the project. The main dependencies are as follows:

```text
python>=3.8,<3.11
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
numpy>=1.23.0
opencv-python>=4.7.0.68
Pillow>=9.5.0
pyyaml>=6.0
tqdm>=4.65.0
matplotlib>=3.6.0
scikit-learn>=1.1.0
pycocotools>=2.0.6
thop>=0.1.1.post2207130030
albumentations>=1.0.3
```

Installation method:

```bash
pip install -r requirements.txt
```

---

## 2. Dataset
The dataset of this project is stored in LKAM_data under Dataset. Among them, images stores the content about the original image, and lables stores the image tags processed by lableImg. Some of its graphics are shown in the figure.

! [img_1.png](img_1.png)
* The COCO dataset can be downloaded at https://cocodataset.org/#home
## 3. How to run the model

All the contents not specified in the following script are executed in the project root directory.

### 33.1 Prepare the dataset


- Put the training/validation images and labels under the 'datasets/LKAM_data/' directory. Example of the directory structure:
```bash
datasets/LKAM_data/
├── images/
│   ├── train/
│   └── val/
└── labels/
├── train/
└── val/
```

Edit 'data.yaml', configure the paths of 'train' and 'val', as well as' nc '(number of categories),' names' (category names), etc.
Specifically,
```bash
train: .\dataset\LKAM_data\images\train  # yolo_coco images (relative to 'path') 128 images
val: .\dataset\LKAM_data\images\val  # val images (relative to 'path') 128 images
test: ./dataset/images/test

# number of classes
nc: 6

# class names
names: ['bird', 'nest', 'plastic', 'toy', 'branch', 'kite']
```
### 3.2 Training
- Run train.py (Note: You can run train.py in pycharm or other compilation software)
```bash
python train.py \
  --model LKAM.yaml \
  --data data.yaml \
  --epochs 300 \
  --batch 16 \
  --imgsz 640 \
  --weights '' \
  --project runs/train \
  --name LKAM_RCatenaryDet
````

* `--model` Specify the base configuration file (which can be replaced by) `yolov8s.yaml`、`yolov8m.yaml` etc）
* `--weights` Empty indicates training from scratch or specifying the pre-training weights `.pt`
*  The result is saved in`runs/train/LKAM_RCatenaryDet/`

* If you want to fine-tune the model on a custom dataset

  * A single graphics card
```bash
python train.py --batch 32  --data data/dataset.yaml --fuse_ab --device 0
````
  * Multiple graphics cards
```bash
python -m torch.distributed.launch --nproc_per_node 8 train.py --batch 256  --data data
````

* If your training terminal is activated, you can restore the previous training process through the following command
  * A single graphics card
```bash
python train.py --resume
````
  * Multiple graphics cards
```bash
python -m torch.distributed.launch --nproc_per_node 8 train.py --resume
````
### Reasoning & Verification

#### Reasoning single picture

```bash
python detect.py \
  --weights runs/train/LKAM_RCatenaryDet/weights/best.pt \
  --source assets/test.jpg \
  --imgsz 640 \
  --conf 0.25 \
  --save-txt --save-conf \
  --project runs/detect \
  --name demo
```
#### Verify the dataset mAP

```bash
python val.py \
  --weights runs/train/LKAM_RCatenaryDet/weights/best.pt \
  --data data/rail.yaml \
  --imgsz 640 \
  --batch 16
```

---

## 4. Core algorithm implementation (pseudo-code)

LKAM-RCatenaryDet adopts a single-stage object detection algorithm. It not only has a fast detection speed, but also improves the ability to capture small objects in images by designing the backbone network LRFSRBackbone that combines the convolution of large convolutional kervolution with heavy parameter optimization and the lightweight module ScK2. Furthermore, a small target detection head combined with GAM was designed to capture the global context information, thereby focusing on the key detection areas. And the Wise-IoU loss function used to improve the convergence of model bounding box prediction and handle class imbalance.
### 4.1 Backbone：LRFSRBackbone

* **Re-parameterized Large-Kernel Fusion**

  * Parallel large convolutional kernels fuse multi-scale features
  * During training, the large kernel and the small kernel branches in parallel. During inference, they are merged into a single equivalent large convolution to improve the speed
   ```bash
  class LRFSRBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 deploy=False,
                 attempt_use_lk_impl=True,
                 with_cp=False,
                 use_sync_bn=False,
                 ffn_factor=4):
        super().__init__()
        self.with_cp = with_cp
        self.need_contiguous = (not deploy) or kernel_size >= 7

        if kernel_size == 0:
            self.dwconv = nn.Identity()
            self.norm = nn.Identity()
        elif deploy:
            self.dwconv = get_conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=True,
                                     attempt_use_lk_impl=attempt_use_lk_impl)
            self.norm = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              use_sync_bn=use_sync_bn,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
        elif kernel_size == 1:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=1, bias=deploy)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
        else:
            assert kernel_size in [3, 5]
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=dim, bias=deploy)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)

        self.se = SEBlock(dim, dim // 4)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim),
                NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                get_bn(dim, use_sync_bn=use_sync_bn))

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if (not deploy) and layer_scale_init_value is not None \
                                                         and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, inputs):

        def _f(x):
            if self.need_contiguous:
                x = x.contiguous()
            y = self.se(self.norm(self.dwconv(x)))
            y = self.pwconv2(self.act(self.pwconv1(y)))
            if self.gamma is not None:
                y = self.gamma.view(1, -1, 1, 1) * y
            return self.drop_path(y) + x

        if self.with_cp and inputs.requires_grad:
            return checkpoint.checkpoint(_f, inputs)
        else:
            return _f(inputs)

    def reparameterize(self):
        if hasattr(self.dwconv, 'merge_dilated_branches'):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, 'running_var') and hasattr(self.dwconv, 'lk_origin'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
            self.dwconv.lk_origin.bias.data = self.norm.bias + (
                        self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            self.norm = nn.Identity()
        if self.gamma is not None:
            final_scale = self.gamma.data
            self.gamma = None
        else:
            final_scale = 1
        if self.act[1].use_bias and len(self.pwconv2) == 3:
            grn_bias = self.act[1].beta.data
            self.act[1].__delattr__('beta')
            self.act[1].use_bias = False
            linear = self.pwconv2[0]
            grn_bias_projected_bias = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
            bn = self.pwconv2[2]
            std = (bn.running_var + bn.eps).sqrt()
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = linear.weight * (bn.weight / std * final_scale).view(-1, 1)
            linear_bias = 0 if linear.bias is None else linear.bias.data
            linear_bias += grn_bias_projected_bias
            new_linear.bias.data = (bn.bias + (linear_bias - bn.running_mean) * bn.weight / std) * final_scale
            self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])
### 4.2 Lightweight ScK2 module
* **ScK2**
  * Group depth-separable convolution + gated reconstruction to achieve spatial redundancy suppression
  * Efficiently aggregate local context information to enhance the representation of small targets
  ```bash
  class SpatialReconstructionUnit(nn.Module):

    def __init__(self, channels, group_num=4, gate_reduction=16):
        super().__init__()
        assert channels % group_num == 0, "channels must be divisible by group_num"
        self.group_num = group_num
        group_ch = channels // group_num

        self.dw_convs = nn.ModuleList([
            nn.Conv2d(group_ch, group_ch, kernel_size=3, padding=1,
                      groups=group_ch, bias=False)
            for _ in range(group_num)
        ])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // gate_reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // gate_reduction, group_num, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        split_x = x.chunk(self.group_num, dim=1) 
        weights = self.gate(x)                     
        outs = []
        for i, (xi, conv) in enumerate(zip(split_x, self.dw_convs)):
            wi = weights[:, i:i+1]                   
            outs.append(conv(xi) * wi)              
        return torch.cat(outs, dim=1)                  

    class ChannelReconstructionUnit(nn.Module):
      def __init__(self, channels, reduction=16):
          super().__init__()
          self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
          self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
          self.act = nn.ReLU(inplace=True)
          self.sigmoid = nn.Sigmoid()
      def forward(self, x):
          y = F.adaptive_avg_pool2d(x, 1)  # (B,C,1,1)
          y = self.act(self.fc1(y))
          y = self.sigmoid(self.fc2(y))
          return x * y                    
    class ScK2(nn.Module):
      
        def __init__(self, channels, group_num=4, gate_reduction=16, cru_reduction=16):
            super().__init__()
            self.sru = SpatialReconstructionUnit(channels, group_num, gate_reduction)
            self.cru = ChannelReconstructionUnit(channels, cru_reduction)
            self.act = nn.ReLU(inplace=True)
            
        def forward(self, x):
            res = x
            x = self.sru(x)
            x = self.cru(x)
            return self.act(x + res)

### 4.3 Head：Global Attention Mechanism（GAM）

* **Global Attention Detection Head**

  * After multi-scale feature fusion, a self-attention layer is added
  * Adaptively adjust the spatial position weight to highlight the target area
   ```bash
  class GlobalAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
            nn.LayerNorm([in_channels // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False)
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        mask = self.conv_mask(x)                
        mask = mask.view(b, 1, h * w)     
        mask = self.softmax(mask)           

        feat_flat = x.view(b, c, h * w)       
        context = torch.einsum('b1n, bcn -> b c 1', mask, feat_flat)  

        transformed = self.transform(context) 
        out = x + transformed                 

        return out
### 4.4 Loss function：Wise-IoU

  * Increase the weight of width difference to enhance the regression accuracy of bounding boxes
   ```bash
     class WiseIoULoss(nn.Module):
      def __init__(self, alpha: float = 0.5, reduction: str = 'mean', eps: float = 1e-7):
          super().__init__()
          self.alpha = alpha
          self.reduction = reduction
          self.eps = eps
  
      def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  
          iou = bbox_iou(pred, target, format='xywh', eps=self.eps)  
          w_pred = pred[..., 2]
          w_true = target[..., 2]
          width_weight = torch.abs(w_pred - w_true) / (torch.max(w_pred, w_true) + self.eps)
  
          wise_iou = iou - self.alpha * width_weight

          loss = 1.0 - wise_iou
  
          if self.reduction == 'mean':
              return loss.mean()
          elif self.reduction == 'sum':
              return loss.sum()
          return loss
   ```
### 4.5 Training Skills

* Data Augmentation  : MixUp, Mosaic, HSV color jitter, etc
* Learning Rate Scheduling  : Cosine annealing + Linear Warm-Up
* Positive and Negative Sample Balance  : Adjust the confidence weight to reduce background false positives
