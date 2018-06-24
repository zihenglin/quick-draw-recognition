# quick-draw-recognition

Quick-draw Doodle recognition. Use Tensorflow object detection on Google Quick Draw data set. Classify and locate your doodles.

![Alt text](media/demo2.gif?raw=true )


### Get started
1. To get started, first download some Google Quick Draw data set from Google Big Query public tables. 

    ```
    cd {YOUR_IMAGE_BASE_DIR}
    wget https://storage.googleapis.com/quickdraw_dataset/full/raw/airplane.ndjson
    wget https://storage.googleapis.com/quickdraw_dataset/full/raw/car.ndjson
    wget https://storage.googleapis.com/quickdraw_dataset/full/raw/apple.ndjson
    wget https://storage.googleapis.com/quickdraw_dataset/full/raw/flower.ndjson
    wget https://storage.googleapis.com/quickdraw_dataset/full/raw/fish.ndjson
    ```
    The entire Quick Draw data set is available [here](https://github.com/googlecreativelab/quickdraw-dataset).

2. Covert Quick Draw data into single images

    ```
    python convert_ndjson_to_png.py --object_limit=500 --n_processes=1 --image_base_dir={YOUR_IMAGE_BASE_DIR}
    ```
    Change object_limit to number of quick draw samples you would like to convert and save. 

3. Randomly draw single quick draw images bigger canvases and save the  canvases. 

    ```
    python combine_quick_drawings.py --total_images=100000 --image_base_dir={YOUR_IMAGE_BASE_DIR} --output_annotation_dir={YOUR_ANNOTATION_DIR} --output_image_dir={YOUR_OUTPUT_IMAGE_DIR}
    ```

4. Create TF Records for training object detection model.

    ```
    python create_tfrecord.py --combined_image_path={YOUR_COMBINED_IMAGE_PATH} --annotation_file_path={YOUR_ANNOTATION_FILE_PATH} --tf_record_output_file_path={YOUR_TF_RECORD_OUTPUT_FILE_PATH}
    ```
    combined_image_path should be the same as output_image_dir in previous step. annotation_file_path should be the same as output_annotation_dir plus the file name.

5. Train object detection model.
    Change the necessary path in train_faster_rcnn_inception_v2.sh and run the file.

    ```
    sh train_faster_rcnn_inception_v2.sh
    ```

6. Run recognition. 

    To run recognition please refer to the [example](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb) in Tensorflow repository.

### Training object detection model
In this project, Faster RCNN with inception network was used. The batch size is set as 12 and the model was trained for about 65k iterations. It take about 15 hours to train from random initialization using a GTX1080 GPU. 

![Alt text](media/total_loss.png?raw=true)




### More demos. Have Fun!
![Alt text](media/demo1.gif?raw=true)


### Have questions?
Feel free to contact me!