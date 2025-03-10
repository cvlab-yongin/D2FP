# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_d2fp_config(cfg):
    """
    Add config for D2FP.
    """
    # NOTE: configs from original D2FP
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "d2fp_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # d2fp model config
    cfg.MODEL.D2FP = CN()

    # loss
    cfg.MODEL.D2FP.DEEP_SUPERVISION = True
    cfg.MODEL.D2FP.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.D2FP.CLASS_WEIGHT = 1.0
    cfg.MODEL.D2FP.DICE_WEIGHT = 1.0
    cfg.MODEL.D2FP.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.D2FP.NHEADS = 8
    cfg.MODEL.D2FP.DROPOUT = 0.1
    cfg.MODEL.D2FP.DIM_FEEDFORWARD = 2048
    cfg.MODEL.D2FP.ENC_LAYERS = 0
    cfg.MODEL.D2FP.DEC_LAYERS = 6
    cfg.MODEL.D2FP.PRE_NORM = False

    cfg.MODEL.D2FP.HIDDEN_DIM = 256
    cfg.MODEL.D2FP.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.D2FP.NUM_PRIORS = 100

    cfg.MODEL.D2FP.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.D2FP.ENFORCE_INPUT_PROJ = False

    cfg.MODEL.D2FP.WITH_HUMAN_INSTANCE = True

    # D2FP inference config
    cfg.MODEL.D2FP.TEST = CN()
    cfg.MODEL.D2FP.TEST.SEMANTIC_ON = True
    cfg.MODEL.D2FP.TEST.PARSING_ON = False
    cfg.MODEL.D2FP.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.D2FP.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.D2FP.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.D2FP.TEST.PARSING_INS_SCORE_THR = 0.5
    cfg.MODEL.D2FP.TEST.METRICS = ("mIoU", "APr", "APp")

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.D2FP.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # Vision Transformer (ViT) backbone config
    cfg.MODEL.VIT = CN()
    cfg.MODEL.VIT.IMG_SIZE = 512
    cfg.MODEL.VIT.PATCH_SIZE = 16
    cfg.MODEL.VIT.EMBED_DIM = 768    # 768 for base, 1024 for large, 1280 for huge
    cfg.MODEL.VIT.DEPTH = 12    # 12 for base, 24 for large, 32 for huge
    cfg.MODEL.VIT.NUM_HEADS = 12    # 12 for base, 16 for large, 16 for huge
    cfg.MODEL.VIT.MLP_RATIO = 4.0
    cfg.MODEL.VIT.QKV_BIAS = True
    cfg.MODEL.VIT.DROP_PATH_RATE = 0.1  # (MAE-pretraining) 0.1 for base, 0.4 for large, 0.5 for huge
    cfg.MODEL.VIT.INIT_VALUES = None    # layer scale: 0.0001 for deit-3
    cfg.MODEL.VIT.NORM_PRE = False
    cfg.MODEL.VIT.NORM_POST = True
    cfg.MODEL.VIT.SWIGLU = False
    cfg.MODEL.VIT.USE_ABS_POS = True
    cfg.MODEL.VIT.USE_REL_POS = False
    cfg.MODEL.VIT.USE_ACT_CHECKPOINT = False
    cfg.MODEL.VIT.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.VIT.PRETRAIN_USE_CLS_TOKEN = True
    cfg.MODEL.VIT.LR_DECAY_RATE = 1.0
    cfg.MODEL.VIT.XFORMERS = True
    cfg.MODEL.VIT.FREEZE = False

    # LoRA config
    cfg.MODEL.LORA = CN()
    cfg.MODEL.LORA.LORA_BLOCK_INDEXES = []
    cfg.MODEL.LORA.LORA_ALPHA = 1.0
    cfg.MODEL.LORA.LORA_DIM = 8

    # ToMe config
    cfg.MODEL.TOME = CN()
    cfg.MODEL.TOME.RATIO = 0.5
    cfg.MODEL.TOME.SX = 2
    cfg.MODEL.TOME.SY = 2
    cfg.MODEL.TOME.USE_RAND = True
    cfg.MODEL.TOME.MERGE_ATTN_INDEXES = []

    # RepAdapter config
    cfg.MODEL.REP_ADAPTER = CN()
    cfg.MODEL.REP_ADAPTER.ADAPTER_BLOCK_INDEXES = []
    cfg.MODEL.REP_ADAPTER.ADAPTER_HIDDEN_DIM = 8
    cfg.MODEL.REP_ADAPTER.ADAPTER_SCALE = 1.0
    cfg.MODEL.REP_ADAPTER.ADAPTER_GROUPS = 2
    cfg.MODEL.REP_ADAPTER.ADAPTER_DROPOUT = 0.1
    
    # simple feature pyramid
    cfg.MODEL.SFP = CN()
    cfg.MODEL.SFP.SFP_DIM = 256
    cfg.MODEL.SFP.SCALE_FACTORS = [4.0, 2.0, 1.0, 0.5]

    # transformer module
    cfg.MODEL.D2FP.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.INPUT.ROTATION = 0

    # Single human parsing aug
    cfg.INPUT.SINGLE_HUMAN = CN()
    cfg.INPUT.SINGLE_HUMAN.ENABLED = False
    cfg.INPUT.SINGLE_HUMAN.SIZES = ([384, 512],)
    cfg.INPUT.SINGLE_HUMAN.SCALE_FACTOR = 0.8
    cfg.INPUT.SINGLE_HUMAN.ROTATION = 40
    cfg.INPUT.SINGLE_HUMAN.COLOR_AUG_SSD = False
    cfg.INPUT.SINGLE_HUMAN.TEST_SCALES = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.D2FP.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.D2FP.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.D2FP.IMPORTANCE_SAMPLE_RATIO = 0.75

    # WandB
    cfg.WANDB = CN({"ENABLED": False})
    cfg.WANDB.ENTITY = ""
    cfg.WANDB.NAME = ""
    cfg.WANDB.PROJECT = "D2FP"

    cfg.SEED = 0
