# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_d2fp_config

# dataset loading
from .data.dataset_mappers.d2fp_semantic_hp_dataset_mapper import D2FPSemanticHPDatasetMapper
from .data.dataset_mappers.d2fp_single_human_test_dataset_mapper import D2FPSingleHumanTestDatasetMapper
from .data.dataset_mappers.d2fp_parsing_dataset_mapper import D2FPParsingDatasetMapper
from .data.dataset_mappers.d2fp_parsing_lsj_dataset_mapper import D2FPParsingLSJDatasetMapper
from .data.build import build_detection_test_loader


# models
from .d2fp_model import D2FP
from .test_time_augmentation import SemanticSegmentorWithTTA, ParsingWithTTA

# evaluation
from .evaluation.parsing_evaluation import ParsingEvaluator
from .evaluation.utils import load_image_into_numpy_array

# utils
from .utils.wandb_writer import WandBWriter
