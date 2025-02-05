import coremltools as ct
from coremltools.models.neural_network import quantization_utils

mlmodel = ct.models.MLModel("stdit3.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)

op_config = ct.optimize.coreml.OpPalettizerConfig(
    mode="kmeans",
    nbits=8,
)

config = ct.optimize.coreml.OptimizationConfig(
    global_config=op_config,
    op_type_configs={
        "gather": None # avoid quantizing the embedding table
    }
)

model = ct.optimize.coreml.palettize_weights(mlmodel, config=config).save("quantize/stdit3.mlpackage")