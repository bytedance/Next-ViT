# -*- coding: utf-8 -*-
import torch
import argparse
from timm.models import create_model
import coremltools as ct
import sys
sys.path.append("../classification")
import nextvit
import utils
parser = argparse.ArgumentParser('Next-ViT export CoreML model script', add_help=False)
parser.add_argument(
    '--batch-size',
    type=int,
    default=1,
    help='batch size used to export CoreML model.'
)
parser.add_argument(
    '--image-size',
    type=int,
    default=224,
    help='image size used to export CoreML model.'
)
parser.add_argument(
    '--model',
    type=str,
    help='model type'
)
args = parser.parse_args()

def main():
    model = create_model(
        args.model,
        num_classes=1000,
    )
    model.eval()
    input_tensor = torch.zeros((args.batch_size, 3, args.image_size, args.image_size), dtype=torch.float32)
    utils.cal_flops_params_with_fvcore(model, input_tensor)

    # Merge pre bn before exporting onnx/coreml model to speedup inference.
    if hasattr(model, "merge_bn"):
        model.merge_bn()

    coreml_file = "./%s_%dx%d.mlmodel" % (args.model, args.image_size, args.image_size)
    traced_model = torch.jit.trace(model, input_tensor)

    out = traced_model(input_tensor)
    model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(shape=input_tensor.shape)]
    )
    model.save(coreml_file)
    print("CoreML model saved to: %s."%coreml_file)
if __name__ == '__main__':
    main()
