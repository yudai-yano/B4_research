@echo off

set dir=C:\Users\sk122\inheriting
set Imagepath=%dir%\%1\%3
@REM %1: res\{subject_name}\images\{file_name}\bg_removed, %3: bg_removed_image_0
set Outputdir=%dir%\%2\
@REM %2: res\{subject_name}\alphapose\{file_name}
set cfg=configs\halpe_26\resnet\256x192_res50_lr1e-3_1x.yaml
set checkpoint=pretrained_models\halpe26_fast_res50_256x192.pth
set format=open

cd %dir%
cd AlphaPose
echo Start alphapose to %3...

@REM alphapose process start
python scripts\demo_inference.py --cfg %cfg% --checkpoint %checkpoint% --image %Imagepath%.png --outdir %Outputdir% --format open --vis_fast --save_img