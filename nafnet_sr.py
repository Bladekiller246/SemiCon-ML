"""NAFNet-SR: NAFNet body at LR resolution + PixelShuffle x2 head.

Self-contained (torch only, no basicsr import) so that evaluate.py runs on a
fresh machine with nothing but `pip install torch numpy`.

Architecture notes
------------------
The official NAFNet is same-resolution in/out. Our task is 128 -> 256, so:
  * the UNet body runs at 128x128  -> 4x cheaper than upsampling first
  * a PixelShuffle(2) head lifts width -> 1*(2^2) channels -> 1 channel at 256
  * the global residual is a bicubic upsample of the input
Everything inside the blocks is byte-for-byte the official NAFBlock, so official
NAFNet checkpoints load into the body (see load_pretrained_body).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over (C,) for NCHW tensors. Matches basicsr."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Official NAFNet block: LN -> 1x1 -> 3x3 dw -> SimpleGate -> SCA -> 1x1,
    then LN -> 1x1 -> SimpleGate -> 1x1. No nonlinear activations."""

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1,
                               groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # skip-init: blocks start as identity, which is what lets NAFNet train
        # at lr=1e-3 without warmup blowups.
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class UpsampleConv(nn.Module):
    """Resize-then-convolve upsampler (Odena et al.), a drop-in replacement for
    `Conv1x1 -> PixelShuffle(2)`.

    PixelShuffle produces scale^2 separate channels and interleaves them into a
    2x2 tile. Nothing constrains those channels to agree, so training can drive
    them apart and the interleaving bakes the disagreement into a fixed periodic
    pattern -- MEASURED at every internal decoder stage of the trained model
    (amplitude 2.257 / 0.179 / 0.044 / 0.286 on a constant, zero-noise input,
    where a translation-equivariant network must output a constant).

    Upsampling first removes that degree of freedom: interpolation is a fixed
    smooth operation with no learnable per-sub-pixel weights, and the conv that
    follows sees a full-resolution feature map.

    !!! DO NOT USE THIS FOR THE INTERNAL DECODER UPSAMPLERS (up_mode='resize').
    MEASURED, controlled A/B against an otherwise identical run:

        up_mode=pixelshuffle (a_both)   25.189 dB / 0.6114 SSIM
        up_mode=resize       (up_resize) 16.098 dB / 0.3769 SSIM   <-- 9 dB WORSE

    Why: PixelShuffle is information-PRESERVING. It is a pure reshape that
    trades channels for space and loses nothing, so the decoder can express
    genuine high-frequency detail by placing different values in each sub-pixel.
    Bilinear interpolation is a LOW-PASS FILTER; applying it at all four decoder
    stages progressively destroys the high-frequency content this task exists to
    reconstruct. The trained model collapsed to a washed-out output (std 0.134
    vs ground truth 0.236, range floor stuck at 0.200 instead of 0.0).

    The sub-pixel freedom that lets PixelShuffle emit a checkerboard is the SAME
    mechanism it uses to represent fine detail. Removing it costs far more than
    the artifact it prevents.

    This class is retained because it IS correct for the FINAL output head
    (head='resize_conv'), where it measured +0.59 dB and 6000x less checkerboard
    -- there it maps features to a single image channel and there is no
    downstream stage that needs the high-frequency content.
    """

    def __init__(self, c_in, c_out, k=3, mode='bilinear'):
        super().__init__()
        self.mode = mode
        self.conv = nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False)

    def forward(self, x):
        if self.mode == 'nearest':
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.mode, align_corners=False)
        return self.conv(x)


def icnr_(tensor, scale, gain=0.1, initializer=nn.init.kaiming_normal_):
    """ICNR init for the sub-pixel conv: the `scale^2` output planes start as
    copies of one kernel, so PixelShuffle begins free of checkerboard bias.

    `gain` shrinks the init so the network starts close to "output = bicubic
    upsample of input" and learns the correction from there, rather than
    starting with a loud random residual on top of the base.
    """
    out_c, in_c, kh, kw = tensor.shape
    sub = torch.zeros(out_c // (scale ** 2), in_c, kh, kw)
    initializer(sub)
    sub = sub.repeat_interleave(scale ** 2, dim=0) * gain
    with torch.no_grad():
        tensor.copy_(sub)


class NAFNetSR(nn.Module):
    """NAFNet UNet at LR resolution with a x`scale` sub-pixel output head.

    Args:
        img_channel: 1 for this task (grayscale SEM).
        in_mean/in_std: fixed affine applied to the input. The degraded input
            legitimately exceeds [0,1] (speckle overshoot) -- we do NOT
            per-image renormalise it, because its min/max are noise-driven and
            therefore unstable. A fixed shift/scale just centres it.
        res_mode: interpolation used for the global residual base.
    """

    def __init__(self, img_channel=1, width=32, middle_blk_num=12,
                 enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2),
                 scale=2, in_mean=0.45, in_std=0.25, res_mode='bicubic',
                 drop_out_rate=0., head='pixelshuffle',
                 residual='bicubic', input_transform='affine',
                 up_mode='pixelshuffle', use_refine=False, refine_k=5):
        super().__init__()
        self.scale = scale
        self.res_mode = res_mode
        self.head = head
        self.residual = residual
        self.input_transform = input_transform
        # 'gated': let the network dial the noisy base down if passing it
        # through is hurting. Starts at 1.0 = identical to 'bicubic'.
        self.res_gate = nn.Parameter(torch.ones(1)) if residual == 'gated' else None
        self.register_buffer('in_mean', torch.tensor(float(in_mean)))
        self.register_buffer('in_std', torch.tensor(float(in_std)))

        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            if up_mode in ('pixelshuffle', 'pixelshuffle_icnr', 'pixelshuffle_smooth'):
                # PixelShuffle is kept because it is information-preserving (see
                # UpsampleConv docstring for the 9 dB disaster from replacing it).
                # These variants CONSTRAIN the sub-pixel planes instead:
                #   _icnr   : start all scale^2 planes identical
                #   _smooth : + a 3x3 conv AFTER the shuffle so phase artifacts
                #             can be cancelled without low-passing the features
                c1 = nn.Conv2d(chan, chan * 2, 1, bias=False)
                if up_mode != 'pixelshuffle':
                    icnr_(c1.weight, 2, gain=1.0)
                layers = [c1, nn.PixelShuffle(2)]
                if up_mode == 'pixelshuffle_smooth':
                    smooth = nn.Conv2d(chan // 2, chan // 2, 3, padding=1, bias=False)
                    nn.init.dirac_(smooth.weight)      # start as identity
                    layers.append(smooth)
                self.ups.append(nn.Sequential(*layers))
            else:
                # 'resize' / 'resize_nearest': checkerboard-free by construction
                self.ups.append(UpsampleConv(
                    chan, chan // 2,
                    mode='nearest' if up_mode == 'resize_nearest' else 'bilinear'))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(num)]))

        # --- x2 super-resolution head ---
        # MEASURED: the plain 'pixelshuffle' head develops a severe checkerboard
        # artifact DURING TRAINING -- power at the Nyquist corner reaches ~51000x
        # the spectral median after only 6k iters, against 0.1x for ground truth.
        # ICNR init is not enough: it starts the head checkerboard-free (measured
        # 0.0x at init) but nothing stops training from driving the scale^2
        # sub-pixel planes apart again. The two alternatives below remove the
        # degree of freedom that the artifact lives in.
        if head == 'pixelshuffle':
            self.ending = nn.Conv2d(width, img_channel * scale * scale, 3, 1, 1, bias=True)
            icnr_(self.ending.weight, scale)
            nn.init.zeros_(self.ending.bias)
            self.shuffle = nn.PixelShuffle(scale)
            self.post = nn.Identity()
        elif head == 'resize_conv':
            # Odena et al., "Deconvolution and Checkerboard Artifacts": upsample
            # first, then convolve. No sub-pixel planes exist, so a checkerboard
            # is not representable. Costs one conv at 2x resolution.
            self.ending = nn.Identity()
            self.shuffle = nn.Identity()
            self.post = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)
            nn.init.normal_(self.post.weight, std=0.02)
            nn.init.zeros_(self.post.bias)
        elif head == 'pixelshuffle_smooth':
            # keep sub-pixel upsampling (cheap) but give the network a conv AFTER
            # the shuffle so it can cancel phase-dependent artifacts instead of
            # emitting them straight to the output.
            mid = max(img_channel * 4, width // 2)
            self.ending = nn.Conv2d(width, mid * scale * scale, 3, 1, 1, bias=True)
            icnr_(self.ending.weight, scale)
            nn.init.zeros_(self.ending.bias)
            self.shuffle = nn.PixelShuffle(scale)
            self.post = nn.Conv2d(mid, img_channel, 3, 1, 1, bias=True)
            nn.init.normal_(self.post.weight, std=0.02)
            nn.init.zeros_(self.post.bias)
        else:
            raise ValueError(f'unknown head: {head}')

        if residual == 'none':
            # With no base to add, the head IS the image. The near-zero init
            # used above would start the model outputting black and waste the
            # early iterations climbing out of that; bias it to mean brightness.
            final = self.post if isinstance(self.post, nn.Conv2d) else self.ending
            nn.init.normal_(final.weight, std=0.05)
            nn.init.constant_(final.bias, float(in_mean))

        # OPTIONAL final learnable refinement conv, on the image itself.
        #
        # MEASURED: both known upsampler artifacts survive 100k iterations of
        # training under the existing pixel+SSIM+gradient+FFT loss --
        # resize_conv's period-4 pattern was still 14667x ground-truth level at
        # the end of the 'main' run. Those broadband losses give almost no
        # gradient signal at the specific frequencies involved, because one
        # exact bin out of thousands contributes negligibly to total pixel/SSIM
        # error even though it is enormous RELATIVE to what should be there.
        #
        # This layer gives the network a place to fix it, and train.py pairs it
        # with an explicit loss term (artifact_freq_loss) that gives it a reason
        # to. Dirac (identity) init means it is a no-op at step 0 -- it can only
        # help or do nothing, unlike the internal-upsampler swap that measured
        # a 9 dB regression from removing capacity outright. This is additive
        # capacity, not a replacement.
        #
        # A deterministic FFT notch (evaluate.py::notch_artifacts) remains the
        # guaranteed backstop regardless of whether this layer learns anything
        # useful -- it is applied at inference time independent of training.
        self.use_refine = use_refine
        if use_refine:
            self.refine = nn.Conv2d(img_channel, img_channel, refine_k,
                                    padding=refine_k // 2, bias=True)
            nn.init.dirac_(self.refine.weight)
            nn.init.zeros_(self.refine.bias)

        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        return x

    def forward(self, inp):
        _, _, H, W = inp.shape
        x_in = self.check_image_size(inp)

        if self.input_transform == 'log':
            # VARIANCE-STABILISING TRANSFORM. The noise here is multiplicative
            # speckle: Var = s^2 * x^2, so noise amplitude scales with signal
            # (measured: 217x more artifact in bright regions than dark). A CNN
            # with shared spatial weights has to learn one filter that copes
            # with wildly different local SNR. log() turns multiplicative noise
            # into approximately ADDITIVE, constant-variance noise --
            #     log(x*(1+n)) = log(x) + log(1+n) ~= log(x) + n
            # -- which is the regime convolutional denoisers are actually good
            # at. The shift keeps the argument positive (input reaches -0.10).
            x = torch.log(x_in.clamp_min(-0.20) + 0.25)
            x = (x - (-0.30)) / 0.75          # roughly zero-mean/unit-scale
        else:
            x = (x_in - self.in_mean) / self.in_std
        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        if self.head == 'resize_conv':
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear',
                              align_corners=False)
        else:
            x = self.shuffle(self.ending(x))
        x = self.post(x)

        # GLOBAL RESIDUAL. The base is an upsample of the NOISY input, so it
        # carries every bit of the input noise into the output. That makes
        # "output the base unchanged" a cheap local minimum -- the measured
        # symptom being a model that removed only 22.7% of the input noise.
        # 'none' forces the network to synthesise the clean image outright;
        # 'gated' keeps the base but lets training scale it down.
        if self.residual == 'none':
            x = x.float()
        else:
            base = F.interpolate(x_in.float(), scale_factor=self.scale,
                                 mode=self.res_mode, align_corners=False)
            if self.res_gate is not None:
                base = base * self.res_gate.float()
            x = x.float() + base

        if self.use_refine:
            x = self.refine(x)

        return x[:, :, :H * self.scale, :W * self.scale]


# --------------------------------------------------------------------------- #
# Named configs
# --------------------------------------------------------------------------- #
CONFIGS = {
    # ~fast: for ablations and if throughput becomes the binding constraint
    's': dict(width=32, middle_blk_num=4, enc_blk_nums=(1, 1, 1, 8), dec_blk_nums=(1, 1, 1, 1)),
    # default: the official SIDD/GoPro width32 topology
    'm': dict(width=32, middle_blk_num=12, enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2)),
    # only if VRAM + time allow; official width64 topology
    'l': dict(width=64, middle_blk_num=12, enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2)),
}


def build_model(config='m', **overrides):
    kw = dict(CONFIGS[config])
    kw.update(overrides)
    return NAFNetSR(**kw)


@torch.no_grad()
def load_pretrained_body(model, ckpt_path, verbose=True):
    """Load an official NAFNet checkpoint into the body of NAFNetSR.

    Adapts the 3-channel `intro` by averaging over the RGB input axis (the
    correct grayscale reduction for a linear layer). `ending` is skipped
    entirely -- ours has a different shape because of the sub-pixel head.
    Returns (n_loaded, n_total_body).
    """
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    for key in ('params', 'state_dict', 'model'):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}

    own = model.state_dict()
    loaded, skipped = {}, []
    for k, v in sd.items():
        if k.startswith('ending'):
            skipped.append(k)
            continue
        if k not in own:
            skipped.append(k)
            continue
        if k == 'intro.weight' and v.shape[1] != own[k].shape[1]:
            if own[k].shape[1] == 1 and v.shape[0] == own[k].shape[0]:
                v = v.mean(dim=1, keepdim=True)   # RGB -> gray
            else:
                skipped.append(k)
                continue
        if v.shape != own[k].shape:
            skipped.append(k)
            continue
        loaded[k] = v

    model.load_state_dict(loaded, strict=False)
    if verbose:
        print(f'[pretrained] loaded {len(loaded)}/{len(own)} tensors '
              f'from {ckpt_path} ({len(skipped)} skipped)')
    return len(loaded), len(own)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    for name in CONFIGS:
        m = build_model(name)
        x = torch.randn(1, 1, 128, 128)
        y = m(x)
        print(f'{name}: params={count_params(m)/1e6:.2f}M  {tuple(x.shape)} -> {tuple(y.shape)}')
