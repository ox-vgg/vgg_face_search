VGG Face Search Pipeline
========================

Author:

 + Ernesto Coto, University of Oxford – <ecoto@robots.ox.ac.uk>

Usage
-----

The data-ingestion pipeline uses the same settings as the VGG Face Search Service. Therefore, please go to the `service` directory at the root of the repository and check the `settings.py` file:

 1. Check that the `CUDA_ENABLED` flag is set to `False` if you want to use the CPU or set it to `True` if you want to use the GPU.
 2. Make sure that `DEPENDENCIES_PATH` points to the location of the place where the dependency libraries (e.g. Pytorch_Retinaface) are installed.
 3. Make sure that `DATASET_FEATS_FILE` points to the location of your dataset features file. **IMPORTANT: If you do not have a features file it will be created for you at the end of the data-ingestion pipeline, but if you do have one any new feature data will be ADDED to the previous features file !**.
 4. Only change the rest of the settings if you really know what you are doing.
 5. If you are going to use a GPU for the face-detector, please make sure that Pytorch can access your GPU device.

After you have checked the settings, you can already run the data-ingestion pipeline. However, in some cases it might be needed to specify the location of additional libraries in your system (e.g. when CUDA is not installed in its standard location). If this is your case, for Linux and macOS, edit the `start_pipeline.sh` script and add any additional library paths to the $LD_LIBRARY_PATH variable. If you are doing the data-ingestion using `start_pipeline.bat` you will need to add those paths to your system's environment variables.

Once this is done, you can start the data-ingestion pipeline. Below you will find the syntax of the command for Linux/macOS. For Windows, the syntax is the same but use `start_pipeline.bat`.

    ./start_pipeline.sh input_type input_file dataset_folder output_file

where:

 + `input_type`: should be "video" if you are running the data-ingestion with videos or "images" if you are running the data-ingestion with images. The quotes around "video" and "images" are just for clarity, do not use then when invoking the pipeline script.
 + `input_file`: should be the full path to ONE video if `input_type` is "video". Otherwise, it should be the full path to a text file containing the list of images to ingest. In this last case, the paths inside the text file should be relative to the `dataset_folder`.
 + `dataset_folder`: is the full path to the base folder holding the images of your dataset. If you are ingesting videos, selected frames from the video will be copied to your dataset folder. If you are ingesting images, the images should be already in your `dataset_folder`.
 + `output_file`: is the full path to the output feature file. This parameter is OPTIONAL. If it is not provided, the path to the output file will be taken from the `DATASET_FEATS_FILE` constant in `settings.py`. **Remember that every time the pipeline is executed new features are ADDED to the previous features file !**.

After the data-ingestion is finished, you will need to start/restart the VGG Face Search Service to perform face-searches over the new data.
