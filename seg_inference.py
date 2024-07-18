import os
import torch
import argparse
import numpy as np
import os.path as osp
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torchvision.models as models

from torch.utils.data import DataLoader
from seg_dataloader import COCOSegDataset


parser = argparse.ArgumentParser(description='Simmple Semantic Segmentation Inference')

# Directory
parser.add_argument('--root_dir',       type=str,   help='dataset directory',               default=osp.join(osp.dirname(__file__), 'dataset'))
parser.add_argument('--save_dir',       type=str,   help='model directory for saveing',     default=osp.join(osp.dirname(__file__), 'experiments', 'models'))
parser.add_argument('--pred_dir',       type=str,   help='prediction image directory',      default=osp.join(osp.dirname(__file__), 'experiments', 'prediction'))

# Inference
parser.add_argument('--ckpt_name',      type=str,   help='saved weight file name',          default='0099.pth')
parser.add_argument('--raw_height',     type=int,   help='image raw height size',           default=1080)
parser.add_argument('--raw_width',      type=int,   help='image raw width size',            default=1920)

# Parameter
parser.add_argument('--input_height',   type=int,   help='model input height size ',        default=256)
parser.add_argument('--input_width',    type=int,   help='model input width size ',         default=256)
parser.add_argument('--batch_size',     type=int,   help='input batch size for training ',  default=8)
parser.add_argument('--gpu',            type=int,   help='GPU id to use',                   default=0)

args = parser.parse_args()


def restore_original_size(result, mode='nearest'):
    
    result = F.interpolate(result, size=(args.raw_height, args.raw_width), mode=mode)
    
    return result



def test():
    # GPU 셋팅
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f'cuda:{args.gpu}')
            print("Use GPU: {} for training".format(args.gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    
    if not osp.exists(args.pred_dir):
        os.makedirs(args.pred_dir)
        
        
    coco_dataset = COCOSegDataset(root_dir=args.root_dir, 
                                  input_height=args.input_height, input_width=args.input_width)
    dataloader = DataLoader(coco_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=4,
                            pin_memory=True)
    
    
    # 학습된 모델 불러오기
    model = models.segmentation.fcn_resnet50(num_classes=1)
    
    model_dir = osp.join(args.save_dir, args.ckpt_name)
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint)
    model.eval()
    model.cuda(args.gpu)
    
    with torch.no_grad():
        for step, (sample_image, sample_gt) in enumerate(dataloader):
            
            sample_image   = torch.tensor(sample_image, device=device, dtype=torch.float32)
            sample_gt      = torch.tensor(sample_gt,    device=device, dtype=torch.float32)
            sample_gt      = torch.unsqueeze(sample_gt, dim=1)        

            prediction  = model(sample_image)['out']
            
            # 원래 사이즈로 복구
            prediction  = F.sigmoid(prediction)
            prediction  = torch.where(prediction > 0.5, 1., 0.)
            
            sample_image = restore_original_size(sample_image)
            sample_gt    = restore_original_size(sample_gt)
            prediction   = restore_original_size(prediction)
            
            # segmentation 결과 (array)
            # 변수 prediction_cpu가 모델의 최종 결과.
            prediction_cpu   = prediction.cpu().detach().numpy()
            sample_image_cpu = sample_image.cpu().detach().numpy()
            sample_gt_cpu    = sample_gt.cpu().detach().numpy()
            
            # 그림으로 그리기
            for num, (sp_image, sp_gt, sp_pred) in enumerate(zip(sample_image_cpu, sample_gt_cpu, prediction_cpu)):
                sp_image = np.transpose(sp_image, (1, 2, 0))
                sp_gt    = np.transpose(sp_gt,    (1, 2, 0))
                sp_pred  = np.transpose(sp_pred,  (1, 2, 0))
                
                plt.subplot(1, 3, 1)
                plt.imshow(sp_image)
                plt.title('input')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(sp_gt)
                plt.title('gt')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(sp_pred)
                plt.title('prediction')
                plt.axis('off')
                
                plt.tight_layout(w_pad=1.)
                plt.savefig(osp.join(args.pred_dir, f'{step}-{num}.png'))
                plt.close()


if __name__ == "__main__":
    test()