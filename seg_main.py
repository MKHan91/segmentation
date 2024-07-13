import torch
import argparse
import os.path as osp
import multiprocessing as mp
import torchvision.models as models
import torch.nn.functional as F
import os

from torch.utils.data import DataLoader
from torchvision import transforms
from seg_dataloader import COCOSegDataset
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='Simmple Semantic Segmentation')

# Directory
parser.add_argument('--root_dir',       type=str,   help='dataset directory',               default=osp.join(osp.dirname(__file__), 'dataset'))
parser.add_argument('--save_dir',       type=str,   help='model directory for saveing',     default=osp.join(osp.dirname(__file__), 'experiments', 'models'))
parser.add_argument('--log_dir',        type=str,   help='log directory for tensorboard',   default=osp.join(osp.dirname(__file__), 'experiments', 'logs'))

# Parameter
parser.add_argument('--input_height',   type=int,   help='model input height size ',        default=256)
parser.add_argument('--input_width',    type=int,   help='model input width size ',         default=256)
parser.add_argument('--batch_size',     type=int,   help='input batch size for training ',  default=8)
parser.add_argument('--learning_rate',  type=int,   help='learning rate ',                  default=1e-3)
parser.add_argument('--num_epochs',     type=int,   help='epoch number for training',       default=100)
parser.add_argument('--gpu',            type=int,   help='GPU id to use',                   default=0)

args = parser.parse_args()



def main():
    # 경로 셋팅
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if not osp.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
        
    # GPU 셋팅
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f'cuda:{args.gpu}')
            print("Use GPU: {} for training".format(args.gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    
    # 텐서보드 셋팅
    writer = SummaryWriter(args.log_dir)
    
    coco_dataset = COCOSegDataset(root_dir=args.root_dir, 
                                  input_height=args.input_height, input_width=args.input_width)
    dataloader = DataLoader(coco_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=4,
                            pin_memory=True)
    
    # 모델 로드
    model = models.segmentation.fcn_resnet50(weights_backbone=True, num_classes=1)
    model = model.cuda()

    # 최적화
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    
    # 학습
    for epoch in range(args.num_epochs):
        model.train()
        for step, (sample_image, sample_gt) in enumerate(dataloader):
            optimizer.zero_grad()
            
            sample_image   = torch.tensor(sample_image, device=device, dtype=torch.float32)
            sample_gt      = torch.tensor(sample_gt,    device=device, dtype=torch.float32)
            sample_gt      = torch.unsqueeze(sample_gt, dim=1)
            
            output = model(sample_image)['out']
            loss   = criterion(output, sample_gt)
            loss.backward()
            
            optimizer.step()
            
            # 정확도 계산
            # Binary segmentation 이므로 sigmoid 사용.
            output  = F.sigmoid(output)
            output  = torch.where(output > 0.5, 1., 0.)
            
            intersection = torch.logical_and(sample_gt, output).sum()
            union = torch.logical_or(sample_gt, output).sum()
            iou = intersection / union if union > 0 else torch.tensor(0.0)
            
        print(f"Epoch: {'[':>4}{epoch + 1:>4}/{args.num_epochs}] | loss: {loss:.4f} | accuracy: {iou:.4f}")
        torch.save(model.state_dict(), osp.join(args.save_dir, f"{epoch:04d}.pth"))
        
        # 텐서보드
        idx_random = torch.randint(0, args.batch_size, (1,)).item()
        writer.add_image('Input/Image',   sample_image[idx_random], global_step=epoch)
        writer.add_image('Input/Gt',      sample_gt[idx_random],    global_step=epoch)
        
        writer.add_image('Results/Prediction', output[idx_random],  global_step=epoch)
        
        writer.add_scalar('Results/Loss',      loss,   global_step=epoch)
        writer.add_scalar('Results/Accuracy',  iou,    global_step=epoch)
        
        
if __name__ == "__main__":
    main()