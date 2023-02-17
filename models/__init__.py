from models.vit_pretrained.vision_transformer import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224, \
    vit_large_patch16_224
from models.vit_pretrained.vision_transformer import VisionTransformer, Attention
from models.attention import L2Attention, CoBiLiRAttention
from lib.sngp import Laplace

attn_setting_dict = {
    'tiny': {
        'dim': 192,
        'num_heads': 3
    },
    'small': {
        'dim': 384,
        'num_heads': 6
    },
    'base': {
        'dim': 768,
        'num_heads': 12,
    },
    'large': {
        'dim': 1024,
        'num_heads': 16,
    }
}


def replace_attn(module, vit_variant, new_attn_module, **kwargs):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if isinstance(target_attr, Attention):
            new_attn = new_attn_module(**attn_setting_dict[vit_variant], **kwargs)
            setattr(module, attr_str, new_attn)
    for _, ch in module.named_children():
        replace_attn(ch, vit_variant, new_attn_module)


def vit_backbone(num_classes=0, variant='tiny', attention='DPSA', pretrained=True, sn=False, alpha=100):
    if variant == 'tiny':
        model: VisionTransformer = vit_tiny_patch16_224(pretrained)
    elif variant == 'small':
        model: VisionTransformer = vit_small_patch16_224(pretrained)
    elif variant == 'base':
        model: VisionTransformer = vit_base_patch16_224(pretrained)
    elif variant == 'large':
        model: VisionTransformer = vit_large_patch16_224(pretrained)
    else:
        raise NotImplementedError
    model.reset_classifier(num_classes)

    if attention == 'DPSA':
        replace_attn(model, variant, Attention)
    elif attention == 'L2SA':
        replace_attn(model, variant, L2Attention, sn=sn)
    elif attention == 'COBILIR':
        replace_attn(model, variant, CoBiLiRAttention, alpha=alpha)
    else:
        raise NotImplementedError

    return model


def tgp(num_classes, num_data, batch_size, variant, vit=None, attention='CoBiLiR', **kwargs):
    if vit is None:
        feature_extractor: VisionTransformer = vit_backbone(0, variant, attention, **kwargs)
    else:
        feature_extractor: VisionTransformer = vit

    num_deep_features = attn_setting_dict[variant]['dim']
    num_gp_features = 128
    normalize_gp_features = True
    num_random_features = 1024
    mean_field_factor = 25
    ridge_penalty = 1
    lengthscale = 2

    model = Laplace(
        feature_extractor,
        num_deep_features,
        num_gp_features,
        normalize_gp_features,
        num_random_features,
        num_classes,
        num_data,
        batch_size,
        mean_field_factor,
        ridge_penalty,
        lengthscale
    )

    return model
