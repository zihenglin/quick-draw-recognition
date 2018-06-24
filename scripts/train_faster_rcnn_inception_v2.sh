PYTHONPATH=${PATH_TO_TENSORFLOW_MODELS}/models/research:$PYTHONPATH
PYTHONPATH=${PATH_TO_TENSORFLOW_MODELS}/research/slim:$PYTHONPATH
export PYTHONPATH

python ${PATH_TO_TENSORFLOW_MODELS}/models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}