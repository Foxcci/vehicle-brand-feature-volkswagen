"""VGG-16 pool3/pool4 Grad-CAM; foreground visualization; gamma 0.65."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms

def background_mask(rgb, tolerance=3):
    """Exclude near-border-color pixels connected to any edge.
    Designed for prepared constant-gray backgrounds, NOT a semantic segmenter.
    Enclosed gray windows/plates remain in the foreground.
    """
    a=np.asarray(rgb.convert('RGB'))
    border=np.concatenate([a[0],a[-1],a[:,0],a[:,-1]],axis=0)
    color=np.median(border,axis=0)
    near=(np.max(np.abs(a.astype(np.float32)-color),axis=2)<=tolerance).astype(np.uint8)
    _,labels=cv2.connectedComponents(near,connectivity=8)
    edge=np.unique(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]))
    edge=edge[edge!=0]
    return (~np.isin(labels,edge)).astype(np.uint8),color.astype(int).tolist()

def scale_in_foreground(cam,mask):
    # Retain positive-CAM zero; do not amplify background maxima.
    denominator=float(cam[mask>0].max()) if np.any(mask) else 0.
    if denominator<=1e-12:return np.zeros_like(cam)
    return np.clip(cam/denominator,0,1)*mask

def run(input_dir,output_dir,indices=(1,2,3,4,5,6)):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    src,dst=Path(input_dir),Path(output_dir)
    selected = set(indices)
    names = []
    for path in src.glob('I*_*.png'):
        match = re.fullmatch(r'I(\d+)_[A-Za-z0-9]+_(front|back|rear)', path.stem)
        if match and int(match.group(1)) in selected:
            names.append(path)
    names.sort(key=lambda p:int(p.stem.split('_')[0][1:]))
    numbers=[int(p.stem.split('_')[0][1:]) for p in names]
    if len(numbers)!=len(set(numbers)):
        raise ValueError('Duplicate I indices in the selected inputs')
    if set(numbers)!=selected:
        raise ValueError(f'Missing requested image indices: {sorted(selected-set(numbers))}')
    if not names:raise ValueError('No I<number>_<model>_<view>.png inputs found')
    for name in ['overlays','heatmaps','masks','raw_responses']:(dst/name).mkdir(parents=True,exist_ok=True)
    weights=models.VGG16_Weights.IMAGENET1K_V1
    model=models.vgg16(weights=weights).eval()
    for module in model.modules():
        if isinstance(module,torch.nn.ReLU):module.inplace=False
    tfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    rows=[]
    for path in names:
        original=Image.open(path).convert('RGB')
        # Prepared square inputs remain spatially aligned. Pad other ratios, never crop.
        display=ImageOps.pad(original.convert('L').convert('RGB'),(448,448),method=Image.Resampling.BILINEAR,color=(128,128,128))
        image=display.resize((224,224),Image.Resampling.BILINEAR)
        mask448,bg=background_mask(display)
        mask224=np.array(Image.fromarray(mask448).resize((224,224),Image.Resampling.NEAREST))
        x=tfm(image).unsqueeze(0)
        acts={};handles=[]
        for index in [16,23]:
            def capture(module,inp,out,index=index):acts[index]=out
            handles.append(model.features[index].register_forward_hook(capture))
        try:
            logits=model(x)
            target=int(logits.argmax(1).item())
            probabilities=logits.detach().softmax(1)[0]
            gradients=torch.autograd.grad(logits[0,target],[acts[index] for index in [16,23]])
            cams=[]
            for index,g in zip([16,23],gradients):
                raw=F.relu((g.mean((2,3),keepdim=True)*acts[index]).sum(1,keepdim=True))
                raw_array=raw[0,0].detach().numpy()
                np.save(dst/'raw_responses'/f'{path.stem}_layer{index}.npy',raw_array)
                up=F.interpolate(raw,(224,224),mode='bilinear',align_corners=False)[0,0].detach().numpy()
                cams.append(scale_in_foreground(up,mask224))
        finally:
            for h in handles:h.remove()
        fused=scale_in_foreground(.55*cams[0]+.45*cams[1],mask224)
        np.save(dst/'raw_responses'/f'{path.stem}_display_cam224.npy',fused)
        cam448=cv2.resize(fused,(448,448),interpolation=cv2.INTER_LINEAR)
        cam448=np.power(np.clip(cam448,0,1),0.65)*mask448
        heat=cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(cam448,0,1)*255),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
        base=np.array(display)
        alpha=(0.35+0.45*cam448)[...,None]*mask448[...,None]
        overlay=np.uint8(np.clip((1-alpha)*base.astype(np.float32)+alpha*heat,0,255))
        overlay[mask448==0]=base[mask448==0]  # preserve exact original gray outside vehicle
        pure=heat.copy();pure[mask448==0]=base[mask448==0]
        assert np.array_equal(overlay[mask448==0],base[mask448==0])
        assert np.isfinite(fused).all()
        Image.fromarray(overlay).save(dst/'overlays'/path.name)
        Image.fromarray(pure).save(dst/'heatmaps'/path.name)
        Image.fromarray(mask448*255).save(dst/'masks'/path.name)
        topv,topi=probabilities.topk(5)
        rows.append(dict(file=path.name,target=weights.meta['categories'][target],probability=float(probabilities[target]),
                         target_id=target,foreground_fraction=float(mask448.mean()),zero_cam=bool(fused.max()==0),
                         background_color=str(bg),top5=json.dumps([(weights.meta['categories'][int(i)],float(v)) for v,i in zip(topv,topi)]),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(path.name,rows[-1]['target'],flush=True)
    with (dst/'targets_and_checks.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    settings=dict(torch=torch.__version__,torchvision=torchvision.__version__,weights='IMAGENET1K_V1',layers=[16,23],fusion=[.55,.45],
                  target='per-image top1 ImageNet logit, unchanged',mask='border-connected color tolerance 3; display only',
                  normalization='positive CAM scaled to foreground maximum separately per layer, weighted fusion then foreground scaling',
                  postprocess='gamma=0.65 for display only; no blur, CAM dilation or top-percent selection',gamma=0.65,overlay_alpha='0.35+0.45*display_CAM inside foreground',
                  layer_names=['pool3','pool4'], input_indices=list(indices), model_training=False, limitation='ImageNet targets are not brand/component labels. Mask and gamma affect display, not inference. Check recorded targets and masks.')
    (dst/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',default='outputs/gradcam')
    parser.add_argument('--indices',type=int,nargs='+',default=[1,2,3,4,5,6])
    parser.add_argument('--torch-home',help='Optional existing Torch cache directory')
    args=parser.parse_args()
    if args.torch_home:
        os.environ['TORCH_HOME']=str(Path(args.torch_home).resolve())
    run(args.input,args.output,args.indices)
