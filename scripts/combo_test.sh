set -e
P=./.venv/Scripts/python.exe
C="--config s --iters 6000 --val-every 3000 --log-every 3000 --batch 16 --head resize_conv --residual bicubic --input-transform log --w-fft 0.1"
# all three terms together -- never tested, gradient may offset MS-SSIM's weakness
$P -u train.py --name c_all  $C --w-grad 0.15 --w-ssim 0.10 --ms-ssim
# MS-SSIM at a LOWER weight alongside gradient
$P -u train.py --name c_low  $C --w-grad 0.15 --w-ssim 0.03 --ms-ssim
