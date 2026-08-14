set -e
P=./.venv/Scripts/python.exe
C="--config s --iters 6000 --val-every 3000 --log-every 3000 --batch 16 --head resize_conv --residual bicubic --input-transform log --w-fft 0.1"
# add back ONE term at a time on top of the clean L1+FFT baseline
$P -u train.py --name iso_grad $C --w-grad 0.15 --w-ssim 0.0
$P -u train.py --name iso_ssim $C --w-grad 0.0  --w-ssim 0.10 --ms-ssim
