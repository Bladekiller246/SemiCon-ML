set -e
P=./.venv/Scripts/python.exe
C="--config s --iters 6000 --val-every 3000 --log-every 3000 --batch 16 --head resize_conv --residual none --input-transform log"
$P -u train.py --name f_range  $C --up-mode pixelshuffle       --w-range 1.0
$P -u train.py --name f_icnr   $C --up-mode pixelshuffle_icnr  --w-range 0.0
$P -u train.py --name f_smooth $C --up-mode pixelshuffle_smooth --w-range 0.0
$P -u train.py --name f_both   $C --up-mode pixelshuffle_smooth --w-range 1.0
