import torch
import torch.nn as nn
from torch.nn import functional as F
#from timm.models.layers.weight_init import trunc_normal_
from timm.layers import trunc_normal_
import copy

from models.vit import VisionTransformer, PatchEmbed, resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.attention import get_attn_fn, Attention_LoRA

class ViT(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, attn_fn=Attention_LoRA, n_tasks=10, rank=64):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, attn_fn=attn_fn, n_tasks=n_tasks, rank=rank)

    def forward(self, x, task_id, register_blk=-1, get_feat=False, get_cur_feat=False):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        for i,blk in enumerate(self.blocks):
            x = blk(x, task_id, register_blk==i, get_feat=get_feat, get_cur_feat=get_cur_feat)

        x = self.norm(x)
        
        return x



# def _create_vision_transformer(variant, pretrained=False, **kwargs):
#     if kwargs.get('features_only', None):
#         raise RuntimeError('features_only not implemented for Vision Transformer models.')

#     # NOTE this extra code to support handling of repr size for in21k pretrained models
#     # pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
#     pretrained_cfg = resolve_pretrained_cfg(variant)
#     default_num_classes = pretrained_cfg['num_classes']

#     num_classes = kwargs.get('num_classes', default_num_classes)
#     repr_size = kwargs.pop('representation_size', None)
#     if repr_size is not None and num_classes != default_num_classes:
#         repr_size = None

#     model = build_model_with_cfg(
#         ViT, variant, pretrained,
#         pretrained_cfg=pretrained_cfg,
#         representation_size=repr_size,
#         pretrained_filter_fn=checkpoint_filter_fn,
#         pretrained_custom_load='npz' in pretrained_cfg['url'],
#         **kwargs)
#     return model


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE: resolve_pretrained_cfg có thể trả về dict (timm cũ) hoặc PretrainedCfg object (timm mới)
    pretrained_cfg = resolve_pretrained_cfg(variant)

    # Lấy num_classes & url theo cách tương thích hai đời API
    if isinstance(pretrained_cfg, dict):
        default_num_classes = pretrained_cfg.get('num_classes', 1000)
        cfg_url = pretrained_cfg.get('url', '')
    else:
        default_num_classes = getattr(pretrained_cfg, 'num_classes', 1000)
        cfg_url = getattr(pretrained_cfg, 'url', '')

    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # in21k head-size fix: nếu num_classes khác default thì bỏ repr_size
        repr_size = None

    model = build_model_with_cfg(
        ViT, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load=('npz' in cfg_url),
        **kwargs
    )
    return model

class MANet(nn.Module):

    def __init__(self, args):
        super(MANet, self).__init__()

        model_kwargs = dict(patch_size=16, 
                            embed_dim=768, 
                            depth=12, 
                            num_heads=12, 
                            n_tasks=args["total_sessions"], 
                            rank=args["rank"],
                            attn_fn= get_attn_fn(args['lora_type'])
                            )
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224_in21k', pretrained=True, **model_kwargs)


        self.class_num = args["init_cls"]
        self.classifier_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=False)
            for i in range(args["total_sessions"])
        ])

        for m in self.classifier_pool.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                
        self.numtask = 0


    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image,  task_id=None):
        if task_id is None:
            task_id = self.numtask-1

        image_features = self.image_encoder(image, task_id)
        image_features = image_features[:,0,:]
        return image_features

    def forward(self, image, get_feat=False, get_cur_feat=False, fc_only=False):
        if fc_only:
            fc_outs = []
            
            for ti in range(self.numtask):
                fc_outs.append(1*(F.linear(F.normalize(image, p=2, dim=1),F.normalize(self.classifier_pool[ti].weight, p=2, dim=1))))

            return torch.cat(fc_outs, dim=1)

        # logits = []
        image_features = self.image_encoder(image, task_id=self.numtask-1, get_feat=get_feat, get_cur_feat=get_cur_feat)
        class_tokens = image_features[:,0,:]
        class_tokens = class_tokens.view(class_tokens.size(0),-1)
        
        patch_tokens = image_features[:,1:,:]

        logits = 1*(F.linear(F.normalize(class_tokens, p=2, dim=1),F.normalize(self.classifier_pool[self.numtask-1].weight, p=2, dim=1)))

        return {
            'logits': logits,
            'features': class_tokens,
            'patch_tokens': patch_tokens
        }

    def interface(self, image, task_id = None):
        image_features = self.image_encoder(image, task_id=self.numtask-1 if task_id is None else task_id)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        logits = []

        for head in self.classifier_pool[:self.numtask]:
            logits.append(1*(F.linear(F.normalize(image_features, p=2, dim=1),F.normalize(head.weight, p=2, dim=1))))

        logits = torch.cat(logits,1)
        return logits
 
    def update_fc(self, nb_classes):
        self.numtask += 1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self