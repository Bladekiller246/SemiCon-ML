set -e
P=./.venv/Scripts/python.exe
C="--config s --iters 6000 --val-every 3000 --log-every 3000 --batch 16 --head resize_conv"
$P -u train.py --name a_base  $C --residual bicubic --input-transform affine
$P -u train.py --name a_nores $C --residual none    --input-transform affine
$P -u train.py --name a_gated $C --residual gated   --input-transform affine
$P -u train.py --name a_log   $C --residual bicubic --input-transform log
$P -u train.py --name a_both  $C --residual none    --input-transform log
