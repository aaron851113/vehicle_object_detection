import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from datasets import PascalVOCDataset
from utils import *
from eval import evaluate
import warnings

from tqdm.autonotebook import tqdm
from backbone import EfficientDetBackbone
from efficientdet.loss import FocalLoss
import traceback

warnings.simplefilter("ignore", UserWarning)

print(torch.__version__)
print("torch GPU :",torch.cuda.is_available())
print("torch.cuda.current_device()",torch.cuda.current_device())
print("torch.cuda.get_device_name(0)",torch.cuda.get_device_name(0))

# Data parameters
data_folder = '../data'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = 5  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 16  # batch size
iterations = 80000  # number of iterations to train
workers = 0  # number of workers for loading data in the DataLoader
print_freq = 100  # print training status every __ batches
lr = 1e-4  # learning rate
decay_lr_at = [50000, 80000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-5  # weight decay

cudnn.benchmark = True

####################### Efficientdet Loss ############################
class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss
    
########################################################################

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_use = 1
anchors_scales = '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios = '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'

########################################################################
    
def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at
        
    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder, dim=input_sizes[input_use], split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    test_dataset = PascalVOCDataset(data_folder, dim=input_sizes[input_use], split='test',
                                     keep_difficult=keep_difficult)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    
    
    # Load Model
    model = EfficientDetBackbone(num_classes=n_classes, compound_coef=input_use,
                                 ratios=eval(anchors_ratios), scales=eval(anchors_scales))
    
    # load last weights / initial wieght
    start_epoch = 0
    print('[Info] initializing weights...')
    init_weights(model)
    
    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=False)

    # Move to default device
    model = model.to(device)
    
    # set SGD
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)


    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

    bestmAP = 0.

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              optimizer=optimizer,
              epoch=epoch,
              epochs=epochs)
        mAP = evaluate(test_loader, model)
        # Save checkpoint
        if mAP > bestmAP:
            save_bestcheckpoint(epoch, model, optimizer)
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, optimizer, epoch,epochs):
    
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    epoch_loss = []
    # Batches
    for i, (images, annotations) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        annotations = [a.to(device) for a in annotations]

        # Forward prop.
        cls_loss, reg_loss = model(images, annotations, obj_list=None)
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        
        loss = cls_loss + reg_loss
        if loss == 0 or not torch.isfinite(loss):
            continue


        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Update model
        optimizer.step()
        
        epoch_loss.append(float(loss))

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}/{1}] iter: [{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch,epochs, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    scheduler.step(np.mean(epoch_loss))
    del cls_loss, reg_loss, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()
