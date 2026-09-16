#####
# Author: Joseph Metcalfe
# Original Creation Date: 02/02/2025
# Purpose: Transformer model for multispectral multitemporal crop classification with parallel temporal and spectral band processing
#####

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class FeedForwardPAtteRNS(nn.Module):
    def __init__(self, dim, dropout, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class AttentionPAtteRNS(nn.Module):
    def __init__(self, dim, dropout, heads, dim_head):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3,bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, key_padding_mask=None, attn_bias=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        attn_mask = None

        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(1).float()

        if attn_bias is not None:
            attn_mask = attn_bias if attn_mask is None else attn_mask + attn_bias

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        out = rearrange(out,'b h n d -> b n (h d)')

        return self.to_out(out)

class TransformerPAtteRNS(nn.Module):
    def __init__(self, dim, dropout, depth, heads, dimFF):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AttentionPAtteRNS(dim, dropout, heads = heads, dim_head = dim // heads),
                FeedForwardPAtteRNS(dim, dropout, dimFF)
            ]))
    def forward(self, x, key_padding_mask=None, attn_bias=None):
        for attn, ff in self.layers:
            x = attn(x, key_padding_mask=key_padding_mask, attn_bias=attn_bias) + x
            x = ff(x) + x
        return x

class MaskedLayerNormPAtteRNS(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x, mask):
        # x:    (B*H*W, C, T*p*p)
        # mask: (B*H*W, 1, T*p*p)  True = valid
        float_mask = mask.float()
        count = float_mask.sum(dim=-1, keepdim=True).clamp(min=1)

        mean = (x * float_mask).sum(dim=-1, keepdim=True) / count
        var  = ((x - mean).pow(2) * float_mask).sum(dim=-1, keepdim=True) / count

        x_norm = (x - mean) / (var + self.eps).sqrt()
        x_norm = x_norm * self.weight + self.bias

        # return x_norm
        return x_norm * float_mask  # redundant re-zero masked positions post-norm

class CrossAttentionPAtteRNS(nn.Module):
    def __init__( self, dim, heads, dropout=0.0):
        super().__init__()

        self.heads = heads
        self.head_dim = dim // heads
        self.inner_dim = self.head_dim * heads

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.dropout = dropout

    def forward(self, query, context):
        query = self.norm_q(query)
        context = self.norm_kv(context)

        q = rearrange(self.to_q(query),'b n (h d) -> b h n d',h=self.heads)
        k = rearrange(self.to_k(context),'b n (h d) -> b h n d',h=self.heads)
        v = rearrange(self.to_v(context),'b n (h d) -> b h n d',h=self.heads)

        out = torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout if self.training else 0.0,is_causal=False)
        out = rearrange(out,'b h n d -> b n (h d)')

        return self.to_out(out)


#################################### Models ###############################################


class PAtteRNS(nn.Module):
    """
    A Model for multispectral multitemporal crop classificaiton
    
    """
    def __init__(self, modelConf):
        super().__init__()
        self.temporalLength = modelConf['maxTemporalBands']
        self.spectralBands = modelConf['imgSpectralChannels']
        self.numClasses = modelConf['classCount']
        self.imgSize = modelConf['imgSize']
        self.patchSize = modelConf['patchSize']
        self.temporalDim = modelConf['temporalDim']
        self.spectralDim = modelConf['spectralDim']
        self.spatialDim = modelConf['spatialDim']
        self.temporalDepth = modelConf['transformerTemporalDepth']
        self.spectralDepth = modelConf['transformerSpectralDepth']
        self.spatialDepth = modelConf['transformerSpatialDepth']
        self.temporalHeads = modelConf['temporalHeads']
        self.spectralHeads = modelConf['spectralHeads']
        self.spatialHeads = modelConf['spatialHeads']
        self.temporalFFDim = modelConf['temporalFFDim']
        self.spectralFFDim = modelConf['spectralFFDim']
        self.spatialFFDim = modelConf['spatialFFDim']
        self.dropoutValue = modelConf['dropout']

        # DSDS vs DOY
        self.temporalEmbeddingWindow = modelConf['tempEmbeddingLength']
        if modelConf['dateOrder'] == "DOY":
            self.temporalEmbeddingWindow = 365.0

        # patches
        self.numPatches1D = (self.imgSize//self.patchSize)
        self.numPatches2D = self.numPatches1D ** 2

        # tokens & encodings
        self.temporalCls = nn.Parameter(torch.randn(1, self.numClasses, self.temporalDim))
        self.spectralCls = nn.Parameter(torch.randn(1, self.numClasses, self.spectralDim))
        
        self.temporalPosEmbedding = nn.Embedding(int(self.temporalEmbeddingWindow+1), self.temporalDim) # num days in dataset + 1 to avoid dbz
        self.spectralPosEnconding = nn.Parameter(torch.randn(1, self.spectralBands, self.spectralDim))
        self.spatialPosEncodings = nn.Parameter(torch.randn(1, self.numPatches2D, self.spatialDim))

        # restructurings
        self.patchTemporal = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (c p1 p2)', p1 = self.patchSize, p2 = self.patchSize),
            nn.LayerNorm((self.spectralBands) * (self.patchSize ** 2)),
            nn.Linear((self.spectralBands) * self.patchSize ** 2, self.temporalDim),
        )
        # self.patchSpectral = nn.Sequential( # this is the old version without ability to mask padded temporal bands
        #     Rearrange('b t c (h p1) (w p2) -> (b h w) c (t p1 p2)', p1 = self.patchSize, p2 = self.patchSize),
        #     nn.LayerNorm((self.temporalLength) * (self.patchSize ** 2)),
        #     nn.Linear((self.temporalLength) * self.patchSize ** 2, self.spectralDim),
        # )

        self.patchSpectralRearrange = Rearrange( # this lot is replacement for above
            'b t c (h p1) (w p2) -> (b h w) c (t p1 p2)',
            p1=self.patchSize, p2=self.patchSize
        )
        self.patchSpectralNorm = MaskedLayerNormPAtteRNS(self.temporalLength * (self.patchSize ** 2))
        self.patchSpectralLinear = nn.Linear(
            self.temporalLength * self.patchSize ** 2, 
            self.spectralDim
        )

        self.clsPerPatchRearrange = nn.Sequential(
            Rearrange('(b h w) d -> b (h w) (d)', h = self.imgSize//self.patchSize, w = self.imgSize//self.patchSize),
        )
        self.unpatchSpatial = nn.Sequential(
            nn.Linear(self.spatialDim, (self.patchSize ** 2) * self.numClasses), # expand out by number of classes here in prep for loss function
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c = self.numClasses, h=self.imgSize//self.patchSize, w=self.imgSize//self.patchSize, p1=self.patchSize, p2=self.patchSize),
        )

        # dimension used for fusion attention
        self.fusionDim = self.temporalDim # all dims the same now

        # project temporal and spectral CLS to same dim
        self.tempToFuse = nn.Linear(self.temporalDim, self.fusionDim) # all dims the same now
        self.specToFuse = nn.Linear(self.spectralDim, self.fusionDim) # all dims the same now

        # cross attention
        self.crossAttn = CrossAttentionPAtteRNS(
            dim=self.fusionDim,
            heads=self.temporalHeads,
            dropout=self.dropoutValue
        )

        # update spatial embed to reflect new flattened size
        self.combinedClsSpatialEmbed = nn.Sequential(
            nn.Linear((self.numClasses * self.fusionDim) + (self.numClasses * self.fusionDim), self.spatialDim),

        )
        
        # transformers
        self.temporalTransformer = TransformerPAtteRNS(self.temporalDim, self.dropoutValue, self.temporalDepth, self.temporalHeads, self.temporalFFDim)
        self.spectralTransformer = TransformerPAtteRNS(self.spectralDim, self.dropoutValue, self.spectralDepth, self.spectralHeads, self.spectralFFDim)
        self.spatialTransformer = TransformerPAtteRNS(self.spatialDim, self.dropoutValue, self.spatialDepth, self.spatialHeads, self.spatialFFDim)

        self.normTemp = nn.LayerNorm(self.temporalDim)
        self.normSpec = nn.LayerNorm(self.spectralDim)
        self.normFuse = nn.LayerNorm(self.fusionDim)
        
        self.dropout = nn.Dropout(self.dropoutValue)

        self.expand = max(32, self.numClasses * 4)
        self.refineHead = nn.Sequential(
            nn.Conv2d(self.numClasses, self.expand, 3, padding=1, bias=True),
            nn.BatchNorm2d(self.expand),
            nn.ReLU(),
            nn.Conv2d(self.expand, self.expand, 3, padding=1, bias=True),
            nn.BatchNorm2d(self.expand),
            nn.ReLU(),
            nn.Conv2d(self.expand, self.numClasses, 1)
        )

    def forward(self, x):
        B, T, C, hImg, wImg = x.shape
        H, W = hImg//self.patchSize, wImg//self.patchSize

        # take out the temporal info band and convert them to longs for embedding
        dateBand = x[:, :, 0, 0, 0]
        dateBand = (dateBand * self.temporalEmbeddingWindow).long().clamp(0, int(self.temporalEmbeddingWindow))

        # padded temporal band attn masking
        t_mask = (x[:, :, 1:, :, :] == 0).all(dim=2).all(dim=2).all(dim=2)
        t_mask = (t_mask.unsqueeze(1).expand(-1, H*W, -1).reshape(B*H*W, T))
        cls_mask = torch.zeros(B*H*W, self.numClasses, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, t_mask],dim=1)

        p2 = self.patchSize ** 2
        spec_mask = t_mask.unsqueeze(-1)                          # (B*H*W, T, 1)
        spec_mask = spec_mask.expand(-1, -1, p2)                  # (B*H*W, T, p*p)
        spec_mask = spec_mask.reshape(B*H*W, 1, T * p2)           # (B*H*W, 1, T*p*p)

        # remove temporal band from spectral channels
        x = x[:, :, 1:, :, :]

        # embedding into B*H*W, T or C, D; this happens for both early to free up memory
        xTemp = self.patchTemporal(x)
        # xSpec = self.patchSpectral(x)
        xSpec = self.patchSpectralRearrange(x)     # (B*H*W, C, T*p*p)
        xSpec = self.patchSpectralNorm(xSpec, ~spec_mask) # inverted mask
        xSpec = self.patchSpectralLinear(xSpec)                   # (B*H*W, C, spectralDim)

        del x

        # temporal transformer sequence
        # embedding into B*H*W, T, D
        # put temporal pos encodings back in
        tempPosEnc = self.temporalPosEmbedding(dateBand)    # (B, T, D)
        tempPosEnc = tempPosEnc.unsqueeze(1).repeat(1, H*W, 1, 1)
        tempPosEnc = tempPosEnc.view(B*H*W, T, self.temporalDim)
        xTemp += tempPosEnc
        # construct and add class token
        tempCls = self.temporalCls.expand((B*H*W), -1, -1)
        xTemp = torch.cat([tempCls, xTemp], dim = 1)
        # dropout and into transformer
        xTemp = self.dropout(xTemp)
        xTemp = self.temporalTransformer(xTemp, key_padding_mask=full_mask)
        # cut out all but cls tokens for fusion
        xTemp = self.normTemp(xTemp[:, :self.numClasses, :])

        # spectral transformer sequence
        # add positional encodings
        xSpec += self.spectralPosEnconding
        # construct and add class token
        specCls = self.spectralCls.expand((B*H*W), -1, -1)
        xSpec = torch.cat([specCls, xSpec], dim = 1)
        # dropout and into transformer
        xSpec = self.dropout(xSpec)
        xSpec = self.spectralTransformer(xSpec)
        # cut out all but cls tokens for fusion
        xSpec = self.normSpec(xSpec[:, :self.numClasses, :])

        # cross attention: spectral queries temporal
        xSpec_attn = self.crossAttn(query=xSpec,context=xTemp) # sdpa implementation
        xSpec_attn = self.normFuse(xSpec_attn)

        # flatten
        xSpat = torch.cat([xTemp.flatten(1),xSpec_attn.flatten(1)], dim=1)

        del xTemp
        del xSpec
        del xSpec_attn

        # split out to B, H*W, d+d
        xSpat = self.clsPerPatchRearrange(xSpat)
        # project the concated spatial dimension to spatial dim value
        xSpat = self.combinedClsSpatialEmbed(xSpat)
        # spatial transformer sequence
        # add positional encodings, then go through dropout and into the spatial transformer
        xSpat += self.spatialPosEncodings
        xSpat = self.dropout(xSpat)
        xSpat = self.spatialTransformer(xSpat)

        # reshape back out into patches and from there pixels, dimension 1 becomes = n classes here, also essentially where data moves from XD to 2D
        xSpat = self.unpatchSpatial(xSpat)
        xSpat = xSpat + self.refineHead(xSpat) # CRH

        return xSpat



####################################################################################################
#############################              Ablation Model             ##############################
####################################################################################################


class PAtteRNSArchitectureAblation(nn.Module):
    """
    A Model for multispectral multitemporal crop classificaiton
    Ablation study options for toggling transformers, cross attention, refine head
    also includes dim size, num cls tokens, pad masking
    date encoding and patch size inherent config options
    """
    class ZeroResidual(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)
        
    def _fuseSimple(self, xTemp, xSpec):
        return torch.cat([xTemp.flatten(1), xSpec.flatten(1)], dim=1)

    def _fuseSqT(self, xTemp, xSpec):
        xSpec_attn = self.crossAttn(query=xSpec, context=xTemp)
        xSpec_attn = self.normFuse(xSpec_attn)
        return torch.cat([xTemp.flatten(1), xSpec_attn.flatten(1)], dim=1)

    def _fuseTqS(self, xTemp, xSpec):
        xTemp_attn = self.crossAttn(query=xTemp, context=xSpec)
        xTemp_attn = self.normFuse(xTemp_attn)
        return torch.cat([xTemp_attn.flatten(1), xSpec.flatten(1)], dim=1)

    def _fuseBi(self, xTemp, xSpec):
        xSpec_attn = self.crossAttnST(query=xSpec, context=xTemp)
        xTemp_attn = self.crossAttnTS(query=xTemp, context=xSpec)
        xSpec_out = self.normFuseSpec(xSpec + xSpec_attn)
        xTemp_out = self.normFuseTemp(xTemp + xTemp_attn)
        return torch.cat([xTemp_out.flatten(1), xSpec_out.flatten(1)], dim=1)

    def __init__(self, modelConf):
        super().__init__()
        self.temporalLength = modelConf['maxTemporalBands']
        self.spectralBands = modelConf['imgSpectralChannels']
        self.numClasses = modelConf['classCount']
        self.imgSize = modelConf['imgSize']
        self.patchSize = modelConf['patchSize']
        self.temporalDim = modelConf['temporalDim']
        self.spectralDim = modelConf['spectralDim']
        self.spatialDim = modelConf['spatialDim']
        self.temporalDepth = modelConf['transformerTemporalDepth']
        self.spectralDepth = modelConf['transformerSpectralDepth']
        self.spatialDepth = modelConf['transformerSpatialDepth']
        self.temporalHeads = modelConf['temporalHeads']
        self.spectralHeads = modelConf['spectralHeads']
        self.spatialHeads = modelConf['spatialHeads']
        self.temporalFFDim = modelConf['temporalFFDim']
        self.spectralFFDim = modelConf['spectralFFDim']
        self.spatialFFDim = modelConf['spatialFFDim']
        self.dropoutValue = modelConf['dropout']

        # ablation configs
        self.useTemporal = modelConf['temporalBranch']  # True, False
        self.useSpectral = modelConf['spectralBranch']  # True, False
        self.useSpatial  = modelConf['spatialBranch']   # True, False

        self.fuseBlockType = modelConf['fuseBlock'] # "Simple", "BiXAttn", "SonTXAttn", "TonSXAttn"
        self.refineHead = modelConf['refineHead'] # True, False

        self.dim256 = modelConf['dim256'] # True, False. Doubles dims and ff dims in model.
        self.kClsTokens = modelConf['kClsTokens'] # True, False. False changes it to 1.
        self.padMasking = modelConf['padMasking'] # True, False. Controls if padded temporal bands are masked in both temporal and spectral transformers.

        # fuse block only meaningful when both temp and spec present
        if not (self.useTemporal and self.useSpectral):
            if self.fuseBlockType != "Simple":
                raise ValueError(
                    f"fuseBlock must be 'Simple' when temporalBranch or spectralBranch is disabled"
                )

        if self.dim256: # dim 128 vs 256 ablation
            self.temporalDim = self.temporalDim * 2
            self.spectralDim = self.spectralDim * 2
            self.spatialDim = self.spatialDim * 2
            self.temporalFFDim = self.temporalFFDim * 2
            self.spectralFFDim = self.spectralFFDim * 2
            self.spatialFFDim = self.spatialFFDim * 2

        self.numCls = 1 # 1 vs k cls tokens ablation
        if self.kClsTokens:
            self.numCls = self.numClasses

        # dimension used for fusion attention
        self.fusionDim = self.temporalDim

        # fusion modules only required when both branches exist
        if self.useTemporal and self.useSpectral:

            if self.fuseBlockType == "Simple":
                pass

            elif self.fuseBlockType == "SqTXAttn":
                self.crossAttn = CrossAttentionPAtteRNS(dim=self.fusionDim, heads=self.temporalHeads, dropout=self.dropoutValue)
                self.normFuse = nn.LayerNorm(self.fusionDim)

            elif self.fuseBlockType == "TqSXAttn":
                self.crossAttn = CrossAttentionPAtteRNS(dim=self.fusionDim, heads=self.temporalHeads, dropout=self.dropoutValue)
                self.normFuse = nn.LayerNorm(self.fusionDim)

            elif self.fuseBlockType == "BiXAttn":
                self.crossAttnST = CrossAttentionPAtteRNS(dim=self.fusionDim, heads=self.temporalHeads, dropout=self.dropoutValue)
                self.crossAttnTS = CrossAttentionPAtteRNS(dim=self.fusionDim, heads=self.temporalHeads, dropout=self.dropoutValue)
                self.normFuseSpec = nn.LayerNorm(self.fusionDim)
                self.normFuseTemp = nn.LayerNorm(self.fusionDim)

            else:
                raise ValueError(f"Unknown fuseBlockType: {self.fuseBlockType}")

        # fuse block function dispatch table 
        self._fuseFns = {
            "Simple":   self._fuseSimple,
            "SqTXAttn": self._fuseSqT,
            "TqSXAttn": self._fuseTqS,
            "BiXAttn":  self._fuseBi,
        }
        self.fuseFn = self._fuseFns[self.fuseBlockType]

        # DSDS vs DOY
        self.temporalEmbeddingWindow = modelConf['tempEmbeddingLength']
        if modelConf['dateOrder'] == "DOY":
            self.temporalEmbeddingWindow = 365.0

        # patches
        self.numPatches1D = (self.imgSize // self.patchSize)
        self.numPatches2D = self.numPatches1D ** 2

        # tokens & encodings
        self.temporalCls = nn.Parameter(torch.randn(1, self.numCls, self.temporalDim))
        self.spectralCls = nn.Parameter(torch.randn(1, self.numCls, self.spectralDim))

        self.temporalPosEmbedding = nn.Embedding(int(self.temporalEmbeddingWindow + 1), self.temporalDim)
        self.spectralPosEnconding = nn.Parameter(torch.randn(1, self.spectralBands, self.spectralDim))
        self.spatialPosEncodings = nn.Parameter(torch.randn(1, self.numPatches2D, self.spatialDim))

        # restructurings
        self.patchTemporal = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (c p1 p2)', p1=self.patchSize, p2=self.patchSize),
            nn.LayerNorm(self.spectralBands * (self.patchSize ** 2)),
            nn.Linear(self.spectralBands * self.patchSize ** 2, self.temporalDim),
        )

        self.patchSpectral = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) c (t p1 p2)', p1=self.patchSize, p2=self.patchSize),
            nn.LayerNorm(self.temporalLength * (self.patchSize ** 2)),
            nn.Linear(self.temporalLength * self.patchSize ** 2, self.spectralDim),
        )

        # used only for the spatial-only ablation
        self.patchSpatial = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b (h w) (t c p1 p2)', p1=self.patchSize, p2=self.patchSize),
            nn.LayerNorm(self.temporalLength * self.spectralBands * (self.patchSize ** 2)),
            nn.Linear(self.temporalLength * self.spectralBands * (self.patchSize ** 2), self.spatialDim),
        )

        self.patchSpectralRearrange = Rearrange('b t c (h p1) (w p2) -> (b h w) c (t p1 p2)', p1=self.patchSize, p2=self.patchSize)

        self.patchSpectralNorm = MaskedLayerNormPAtteRNS(self.temporalLength * (self.patchSize ** 2))
        self.patchSpectralLinear = nn.Linear(self.temporalLength * self.patchSize ** 2, self.spectralDim)

        self.clsPerPatchRearrange = nn.Sequential(
            Rearrange('(b h w) d -> b (h w) d', h=self.imgSize // self.patchSize, w=self.imgSize // self.patchSize),
        )

        # determine fused feature dimension
        if self.useTemporal and self.useSpectral:
            fuseInputDim = 2 * self.numCls * self.fusionDim
        elif self.useTemporal or self.useSpectral:
            fuseInputDim = self.numCls * self.fusionDim
        else:
            # spatial-only never uses fused CLS tokens
            fuseInputDim = self.spatialDim

        self.unpatchSpatial = nn.Sequential(
            nn.Linear(self.spatialDim, (self.patchSize ** 2) * self.numClasses),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=self.numClasses, h=self.imgSize // self.patchSize, w=self.imgSize // self.patchSize, p1=self.patchSize, p2=self.patchSize),
        )

        # only required when spatial transformer is disabled
        if not self.useSpatial:
            self.directUnpatch = nn.Sequential(
                nn.Linear(self.fusionDim, (self.patchSize ** 2) * self.numClasses),
                Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=self.numClasses, h=self.imgSize // self.patchSize, w=self.imgSize // self.patchSize, p1=self.patchSize, p2=self.patchSize),
            )

        # CLS projection only required when CLS tokens exist
        if self.useTemporal or self.useSpectral:
            self.combinedClsSpatialEmbed = nn.Sequential(
                nn.Linear(fuseInputDim, self.spatialDim),
            )
        
        
        # transformers
        self.temporalTransformer = TransformerPAtteRNS(self.temporalDim, self.dropoutValue, self.temporalDepth, self.temporalHeads, self.temporalFFDim)
        self.spectralTransformer = TransformerPAtteRNS(self.spectralDim, self.dropoutValue, self.spectralDepth, self.spectralHeads, self.spectralFFDim)
        self.spatialTransformer = TransformerPAtteRNS(self.spatialDim, self.dropoutValue, self.spatialDepth, self.spatialHeads, self.spatialFFDim)

        self.normTemp = nn.LayerNorm(self.temporalDim)
        self.normSpec = nn.LayerNorm(self.spectralDim)
        # self.normFuse = nn.LayerNorm(self.fusionDim)
        
        self.dropout = nn.Dropout(self.dropoutValue)

        if self.refineHead:
            self.expand = max(32, self.numClasses * 4)
            self.refineHead = nn.Sequential(
                nn.Conv2d(self.numClasses, self.expand, 3, padding=1, bias=True),
                nn.BatchNorm2d(self.expand),
                nn.ReLU(),
                nn.Conv2d(self.expand, self.expand, 3, padding=1, bias=True),
                nn.BatchNorm2d(self.expand),
                nn.ReLU(),
                nn.Conv2d(self.expand, self.numClasses, 1)
            )
        else:
            self.refineHead = self.ZeroResidual()

    def forward(self, x):
        B, T, C, hImg, wImg = x.shape
        H, W = hImg//self.patchSize, wImg//self.patchSize

        # take out the temporal info band and convert them to longs for embedding
        dateBand = x[:, :, 0, 0, 0]
        dateBand = (dateBand * self.temporalEmbeddingWindow).long().clamp(0, int(self.temporalEmbeddingWindow))

        # padded temporal band attn masking
        t_mask = (x[:, :, 1:, :, :] == 0).all(dim=2).all(dim=2).all(dim=2)
        t_mask = (t_mask.unsqueeze(1).expand(-1, H*W, -1).reshape(B*H*W, T))
        cls_mask = torch.zeros(B*H*W, self.numCls, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, t_mask],dim=1)
        if not self.padMasking:
            full_mask = None

        p2 = self.patchSize ** 2
        spec_mask = t_mask.unsqueeze(-1)                          # (B*H*W, T, 1)
        spec_mask = spec_mask.expand(-1, -1, p2)                  # (B*H*W, T, p*p)
        spec_mask = spec_mask.reshape(B*H*W, 1, T * p2)           # (B*H*W, 1, T*p*p)

        # remove temporal band from spectral channels
        x = x[:, :, 1:, :, :]

        # embedding into B*H*W, T or C, D; this happens for both early to free up memory
        # xTemp = self.patchTemporal(x)
        # 
        # del x

        # temporal transformer sequence
        if self.useTemporal:
            # embedding into B*H*W, T, D
            xTemp = self.patchTemporal(x)
            # put temporal pos encodings back in
            tempPosEnc = self.temporalPosEmbedding(dateBand)
            tempPosEnc = tempPosEnc.unsqueeze(1).repeat(1, H*W, 1, 1)
            tempPosEnc = tempPosEnc.view(B*H*W, T, self.temporalDim)
            xTemp += tempPosEnc
            # construct and add class token
            tempCls = self.temporalCls.expand((B*H*W), -1, -1)
            xTemp = torch.cat([tempCls, xTemp], dim=1)
            # dropout and into transformer
            xTemp = self.dropout(xTemp)
            xTemp = self.temporalTransformer(xTemp, key_padding_mask=full_mask)
            # cut out all but cls tokens for fusion
            xTemp = self.normTemp(xTemp[:, :self.numCls, :])
        else:
            xTemp = None

        # spectral transformer sequence
        if self.useSpectral:
            # embedding into B*H*W, C, D
            if self.padMasking:
                xSpec = self.patchSpectralRearrange(x)     # (B*H*W, C, T*p*p)
                xSpec = self.patchSpectralNorm(xSpec, ~spec_mask) # inverted mask
                xSpec = self.patchSpectralLinear(xSpec)                   # (B*H*W, C, spectralDim)
            else:
                xSpec = self.patchSpectral(x)
            # add positional encodings
            xSpec += self.spectralPosEnconding
            # construct and add class token
            specCls = self.spectralCls.expand((B*H*W), -1, -1)
            xSpec = torch.cat([specCls, xSpec], dim=1)
            # dropout and into transformer
            xSpec = self.dropout(xSpec)
            xSpec = self.spectralTransformer(xSpec)
            # cut out all but cls tokens for fusion
            xSpec = self.normSpec(xSpec[:, :self.numCls, :])
        else:
            xSpec = None

        # cross attention / simple fuse swappable block
        if self.useTemporal and self.useSpectral:
            xSpat = self.fuseFn(xTemp, xSpec)

        # single leading transformer:
        elif self.useTemporal:
            xSpat = xTemp.mean(dim=1)
        elif self.useSpectral:
            xSpat = xSpec.mean(dim=1)

        # spatial-only ablation
        else:
            xSpat = self.patchSpatial(x)

        # x is no longer required beyond this point
        del x
        del xTemp, xSpec

        # reshape only when coming from CLS tokens
        if self.useTemporal or self.useSpectral:
            xSpat = self.clsPerPatchRearrange(xSpat)

        if self.useSpatial:
            # project CLS features only when they exist
            if self.useTemporal and self.useSpectral:
                xSpat = self.combinedClsSpatialEmbed(xSpat)

            # spatial transformer sequence
            xSpat += self.spatialPosEncodings
            xSpat = self.dropout(xSpat)
            xSpat = self.spatialTransformer(xSpat)

            # reshape patches back into image
            xSpat = self.unpatchSpatial(xSpat)

        else:
            xSpat = self.directUnpatch(xSpat)
        
        xSpat = xSpat + self.refineHead(xSpat)

        return xSpat