import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

def has_mps() -> bool:
    if not getattr(torch, 'has_mps', False):
        return False
    try:
        torch.zeros(1).to(torch.device("mps"))
        return True
    except Exception:
        return False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if has_mps():
    device=torch.device("mps")


torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Temporal causal filter for Sequencial Images')
parser.add_argument('--inputdir', default="/Volumes/SSD/macstudio/batch_i2i_output")
parser.add_argument('--outputdir',default="/Volumes/SSD/macstudio/batch_i2i_fx")
parser.add_argument('--flicker_times',type=int,default=1)
parser.add_argument('--deblur_times',type=int,default=1)
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

args = parser.parse_args()

try:
    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v3.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
except:
    from model.RIFE import Model
    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded ArXiv-RIFE model")

model.eval()
model.device()

import glob

filenames = [os.path.join(args.inputdir, x) for x in sorted(os.listdir(args.inputdir)) if not x.startswith(".")]
imgfilelist=[file for file in filenames if os.path.isfile(file)]

imgfilelist.sort()

z_1_in=None
z_2_in=None

z_1_out=None
z_2_out=None


for imgfile in imgfilelist:
    print(imgfile)
    img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
    img = (torch.tensor(img.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    n, c, h, w = img.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img = F.pad(img, padding)


    def intp(img1,img2,to_img1_strength=0):
        if id(img1)==id(img2):
            return img1
        if img1 is None:
            return img2
        if img2 is None:
            return img1

        mid_img=model.inference(img1,img2)

        for i in range(to_img1_strength):
            mid_img=model.inference(img1,mid_img)

        return mid_img

    # biquad like process
    z_in_img=intp(z_1_in,z_2_in,0)
    mid_img=intp(img,z_in_img,args.flicker_times) # shallow low pass filter(deflicker)

#    z_out_img=intp(z_1_out,z_2_out,0)
    z_out_img=intp(z_1_out,None,0)
    out_img=intp(mid_img,z_out_img,args.deblur_times) # shallow reverb(deblur)

    # refresh [z]
    z_2_in=z_1_in
    z_1_in=img

    z_2_out = z_1_out
    z_1_out = out_img

    cv2.imwrite(os.path.join(args.outputdir,os.path.basename(imgfile)), (out_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

print("Done")