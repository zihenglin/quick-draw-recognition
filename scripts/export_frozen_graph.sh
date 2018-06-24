PYTHONPATH=${PATH_TO_TENSORFLOW_MODELS}/models/research:$PYTHONPATH
PYTHONPATH=${PATH_TO_TENSORFLOW_MODELS}/research/slim:$PYTHONPATH
export PYTHONPATH

python ${PATH_TO_TENSORFLOW_MODELS}/models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CHECKPOINT_PREFIX} \
    --output_directory=${OUTPUT_DIRECTORY}
