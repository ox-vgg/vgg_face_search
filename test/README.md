vgg_face_search test scripts
===========================

Author:

 + Ernesto Coto, University of Oxford – <ecoto@robots.ox.ac.uk>

#### *About the test script*

As mentioned in the [Usage](https://gitlab.com/vgg/vgg_face_search#usage) section, you need to check the settings of the backend service and run the `start_backend_service.sh` script to start it. Once the backend service is running and has loaded all the dataset files, the script in this folder can be used to test the communication with the backend service. The script will run a face search query, as it is done in [VFF](http://www.robots.ox.ac.uk/~vgg/software/vff/) when using the [vgg_frontend](https://gitlab.com/vgg/vgg_frontend).

However, **the script can also be seen as an example of usage of the vgg_face_search service**, in case you want to develop your own web frontend to replace the [vgg_frontend](https://gitlab.com/vgg/vgg_frontend).

#### *Executing the test script*

First, start the backend service by executing `start_backend_service.sh` as explained in the [Usage](https://gitlab.com/vgg/vgg_face_search#usage) section. Wait until you can see in the terminal the text `FaceRetrieval successfully initialized`, which means the service has loaded all the files associated to face detections in your dataset. Those files must have been produced by running the [ingestion pipeline](https://gitlab.com/vgg/vgg_face_search/tree/master/pipeline).

Then, in a separate terminal window, execute:

```
python test.py <training_images_folder>
```

where `<training_images_folder>` is the full path to a folder with images that contain samples of the face of the person you are looking for in the dataset. Make sure `<training_images_folder>` does not contain special characters and enclose with quotes any path containing blank spaces.

Display the full help of `test.py` with:

```
python test.py -h
```

You will also see other additional arguments that can be useful, such as displaying the results in a GUI, saving the results to a text file, etc.

Please note that all image paths in the results are relative to the folder containing the images when the [ingestion pipeline](https://gitlab.com/vgg/vgg_face_search/tree/master/pipeline) was run. Therefore, when displaying results in a GUI, it might we useful to specify the path to the images folder for actually reading the images. You can do that using the `-p` command-line option of the test script. Make sure the path does not contain special characters and enclose with quotes any path containing blank spaces.
