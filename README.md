# Image Analysis Tutorial

## Quick Links

- [Overview](#overview)
- [Resources](#resources)
- [Step-by-Step Instructions](#step-by-step-instructions)
- [Tutorial Extensions](#tutorial-extensions)
- [Project Ideas](#project-ideas)
- [Cleanup](#cleanup)

## Overview

Medical image processing is a rapidly evolving field that has an important impact on clinical research and practice ([2019 review](https://www.sciencedirect.com/science/article/pii/S093938891830120X) and [2017 review](https://www.annualreviews.org/doi/full/10.1146/annurev-bioeng-071516-044442)). The investigative tasks involved are varied, including detection and registration of cellular entities, image segmentation and classification, and computer-aided diagnosis and outcome prediction. For this project, we have selected a tutorial that mimics a recent [feasibility study](https://www.sciencedirect.com/science/article/pii/S2589750019301086) which applies the Google Cloud AutoML ML Vision tool to train a deep neural network on over 100,000 NIH Chest X-Ray (CXR) images to learn how to predict from among 14 different pathologies. The main purpose of this tutorial is to gain insight into how to evaluate and interpret the quality of blackbox deep learning models. The CXR dataset, Google AutoML tools, and ideas presented in the tutorial may be useful in guiding potential data science projects. While reading the related papers and working through this analysis, you may want to consider the the critical questions for data analysis related to the DSP Course [Competencies](https://cimed-dsp.github.io/competency.html).

## Resources

This tutorial is drawn from a recently published work that demonstrated the potential of new tools that enable medical professionals with no expertise in deep learning image classification techniques to train and deploy sophisticated models.

- Automated deep learning design for medical image classification by health-care professionals with no coding experience: a feasibility study ([link to paper](https://www.sciencedirect.com/science/article/pii/S2589750019301086))

This tutorial, like the paper, uses the Google Cloud AutoML Vision tool, which allows users to train a deep neural network for image classification.

- Documentation for AutoML Vision ([link](https://cloud.google.com/vision/automl/docs/))
- AutoML Vision model description: [blog post](https://ai.googleblog.com/2017/11/automl-for-large-scale-image.html) and [paper](https://arxiv.org/pdf/1707.07012.pdf)

This tutorial will only focus on one of the datasets of the paper, the NIH Chest X-ray (CXR) dataset. This dataset contains 112,120 frontal-view X-ray images for over 30,000 patients. Each image is assigned one or more labels covering 14 different pathologies.

- NIH Chest X-rays: [dataset](https://cloud.google.com/healthcare/docs/resources/public-datasets/nih-chest) and [paper](https://arxiv.org/abs/1705.02315)

## Step-by-Step Instructions

1. Request Access up for Google Cloud Platform
In order to run this tutorial, users will need access to a Google Cloud Platform account and should fill out the [[cloud access request form](https://forms.gle/zBQAxkX1LKoNhi7h9)]. Once your account has been set up, you will be able to create personal Google Cloud Projects and access Google's [AutoML Vision](https://cloud.google.com/vision/automl/docs/) interfaces for automated architecture search and training of convolutional neural network models.

2. Create a Google Cloud Project
All our data and model training will be grouped under one project, which we must first create:

   - Sign into your Google Apps @ Illinois account (set up instruction [here](https://answers.uillinois.edu/illinois/page.php?id=55049)). If you often sign in to a personal Google account, you may want to do this step in an incognito window.
   - Once signed in to Google, you will need to select the project created for you by the uiuc-cimed. Go to the Google Cloud Platform [dashboard](https://console.cloud.google.com/home/dashboard). In the top left corner, you will see the project name listed in the blue bar next to the words 'Google Cloud Platform'. You should confirm that the project is of the form `uiuc-cimed-[netid]`. If not, use the down arrow selector to change the project to the one created for you by uiuc-cimed ('Select from' `Illinois.edu`. on 'All' tab, find `uiuc-cimed-[netid]` and click 'Open')

3. Enable Cloud Storage and AutoML API
In order to complete this tutorial and use AutoML, we must enable some API features that are not enabled by default:

    - Go to the following Enable APIs [link](https://console.cloud.google.com/flows/enableapi?apiid=storage-component.googleapis.com,automl.googleapis.com,storage-api.googleapis.com)
    - Select your `uiuc-cimed-[netid]` project name from drop down list
    - Click 'Continue'
    - We will not need to set up the API credentials at this point, so you do not need to follow the 'Go to credentials' button

4. Create an AutoML Dataset
We will be using AutoML, which is a tool for 'Vision' projects in the Google Cloud Platform 'Artificial Intelligence' suite. To get started with Vision AutoML, we will need to create a Dataset:

    - Go to Vision Datasets page: [https://console.cloud.google.com/vision/datasets](https://console.cloud.google.com/vision/datasets)
    - Select '+ New Dataset'
    - Enter data set name `nih_chest_xray`
    - Select 'Multi-Label Classification'
    - Click 'Create Dataset'

5. Create Cloud Storage Bucket
Next, we need to create a location to store the image data. We will be using a Google Cloud Storage Bucket for this purpose:
    - Go to the Storage dashboard: [https://console.cloud.google.com/storage/](https://console.cloud.google.com/storage/)
    - Click 'Create Bucket'
    - Every bucket needs a unique name, so we will have you modify the name of your bucket with your University [netid]. Name your bucket `nih-chest-xray-bucket-[netid]`, click 'Continue'
    - Chose location type 'Region' and location 'us-central1', click 'Continue'
    - Select storage class 'Standard', click 'Continue'
    - Use defaults for access controls and advanced settings, click 'Create' button

6. Copy X-ray from NIH to personal Bucket:
Now we are going to copy data from the NIH Google Cloud Storage Bucket to our own:
    - Click on the 'Activate Cloud Shell' terminal icon in the top right corner of the platform dashboard *or* activate the shell panel with this [link](http://console.cloud.google.com/?cloudshell=true). If you get a first time message in the Cloud Shell panel at the bottom of the screen, click 'Continue'
    - For working in using the cloud shell in the steps below, we will need to type the commands described and then press the enter key. To get the full identifier for your Google Cloud Project, type in the command line: `gcloud projects list` and press the enter key. The first time you use the Cloud Shell, you may have to 'Authorize' its usage. Find the row with the value `uiuc-cimed-[netid]` in the NAME column, and note the value for the row in the PROJECT_ID column. These instructions will refer to that PROJECT_ID value as [project_id].
    - We can check if we can access the data by listing the first ten images in the NIH chest x-ray image Google Storage Bucket (`gs://gcs-public-data--healthcare-nih-chest-xray/`) using the `uiuc-cimed-[netid]` [project_id] from the previous step. To check, type in the command line:
`gsutil -u [project_id] ls gs://gcs-public-data--healthcare-nih-chest-xray/png | head`
    - To recursively copy the NIH images from their bucket to your Cloud Storage Bucket `gs://nih-chest-xray-bucket-[netid]/`, we can use the command which specifies your project (-u) and the option to copy images in parallel (-m):
`gsutil -u [project_id] -m cp -R gs://gcs-public-data--healthcare-nih-chest-xray/png gs://nih-chest-xray-bucket-[netid]/png`.
      - **Note:** There are ~42 GB of files to copy and the copy command should take up to two hours to complete. Your shell session may time out after an hour, so make sure to check occasionally that it is still active.
      - You can find out more information about the copy command with the commands `gsutil help options` or `gsutil help cp` or the resources [here](https://cloud.google.com/storage/docs/gsutil?hl=ru).
      - Once all of the copying is complete, you should see a operation completion message like:

        ``` bash
        [112.1k/112.1k files][ 42.0 GiB/ 42.0 GiB] 100% Done 0.0    B/s
        Operation completed over 112.1k objects/42.0 GiB.
        ```

      - To check how many files you have copied at the end, try the command: `gsutil -u [project_id] ls -l gs://nih-chest-xray-bucket-[netid]/png | tail -n1` and you should see a message like:

        ``` bash
          TOTAL: 112120 objects, 45057440698 bytes (41.96 GiB)
        ```

7. Upload DataSet CSV
Once we have our own copy of the images, we are ready to upload information about the image labels.
    - Download the image label CSV (already preprocessed for this project), 'nih_chest_xray_full.csv', from [here](https://drive.google.com/open?id=1ct8WvtTZWgOny-Lzw0jM6iyCrWQqyxX-)
      - The original unaltered file with additional detail is available at the NIH [Box](https://nihcc.app.box.com/v/ChestXray-NIHCC) directory: 'Data_Entry_2017.csv'.
      - This preprocessed file has been reformatted using AutoML [guidelines]([https://cloud.google.com/vision/automl/docs/prepare#csv](https://cloud.google.com/vision/automl/docs/prepare#csv))
    - We must modify 'nih_chest_xray_full.csv' to match the name of your unique Google Storage bucket. If you have followed the tutorial instructions, open the 'nih_chest_xray_full.csv' file in your favorite text editor and do a text search and replace, replacing the string 'netid' with your university netid as you typed it in the Google Cloud Storage Bucket name, `nih-chest-xray-bucket-[netid]`
    - Upload the modified 'nih_chest_xray_full.csv' into your Google Storage bucket: [https://console.cloud.google.com/storage/](https://console.cloud.google.com/storage/)
      - Select 'nih-chest-xray-bucket-[netid]'
      - Click 'Upload Files' button
      - Find 'nih_chest_xray_full.csv' and click 'Open'

8. Build AutoML dataset
We are ready to have AutoML preprocess all of the data by moving it into our Dataset and preparing it for model fitting:

    - Go to the AutoML Vision Dataset we created earlier: [https://console.cloud.google.com/vision/datasets/](https://console.cloud.google.com/vision/datasets/)
    - Open the 'nih_chest_xray' dataset
    - On the 'IMPORT' tab, select 'Select a CSV file on Cloud Storage'
    - Click on the 'Browse' button to find the 'nih_chest_xray_full.csv' in the 'nih-chest-xray-bucket-[netid]'. Click 'Select'
    - Once the 'nih_chest_xray_full.csv' file is selected, a green checkbox will appear and you will be able to click 'Continue'.
      - The dataset will begin importing images. **Note:** This may take nearly an hour as the images and labels are pre-processed for training. You will receive an email when the process is complete.

9. Explore NIH chest x-ray Dataset
Once the data is loaded, we can use the AutoML Vision platform to explore the images:

    - Open the 'nih_chest_xray' dataset ([link](https://console.cloud.google.com/vision/datasets/))
    - Click the 'Label Stats' icon near the top. This will tell you the number of images in the dataset for each label as well as how they will be divided into 80% training, 10% validation, and 10% testing (more info [here](https://cloud.google.com/vision/automl/docs/beginners-guide#importing)). Note which classes have the most and the fewest training examples.
    - In the 'Images' tab, select the class that you have the greatest interest in or knowledge of. Browse a few pages of images and click on images to see their full set of labels. Do you disagree with the labels on any examples?

10. Create a new AutoML Vision Model:
In this step, the black box model is trained on the X-ray data.

    - In the 'nih_chest_xray' dataset, click the 'Train' tab to start training an AutoML Vision model
    - You will again see the distribution of the image labels and the partitioning of the images for training and testing
    - Click the 'Start Training' button at the bottom of the page
      - Name your model 'nih_chest_xray_full_data_model'
      - Choose 'Cloud hosted' model, click 'Continue'
      - Change the 'Budget' to 8 node hours
      - **Note:** This process will take many hours to complete as budgeted, however, the interface will provide an estimated completion time. You will receive an email when training is complete.
      - For now, allow "Deploy model to 1 node after training" to remain unselected. Click 'Start Training'

11. Evaluate Model Quality
When training is complete, you will be able to access the model quality statistics and predictions:

    - Go to the AutoML Vision 'Models' dashboard ([link](https://console.cloud.google.com/vision/models))
    - Select the 'nih_chest_xray_full_data_model'
    - At the top of the 'Evaluate' table, you will see a slider to select the confidence threshold that is applied to all predictions. Note its value and read more about model confidence in the AutoML beginner's [guide](https://cloud.google.com/vision/automl/docs/beginners-guide#evaluate)
    - On the left panel, select the most interesting / familiar class label. You will see the performance (precision and recall) graphs for that selected class. Is the current confidence threshold set to the most appropriate level for patient diagnosis? If not, change it to a more appropriate value.
    - In the selected images below, we are shown the A) least confident true positives, B) most confident false negatives, and C) least confident false positives. You can use the navigator arrows to view more examples of each category. Depending on your confidence threshold, you may see the message  `No items to display` instead of any images. For each category, are there examples of images that from your knowledge or opinion are incorrectly labelled or share a pattern?
    - Near the 'Filter labels' text box at the top of the left panel, there are three vertical dots for additional options. Click on this and sort the labels from best performing to worst performing and/or from most to fewest images. What do you observe about class performance? Repeat the previous two steps on a class that performed well and again on a class that performed poorly.
    - Select the 'All labels' class filter in the left panel. You will see a part of the confusion matrix, which shows the percentage of examples from each true row label class that are mislabeled with the predicted column label. To see the entire matrix, you will have to download the table and open in Excel. What classes are being frequently confused? How is the 'No finding' labeling influencing results? What class has the best predictions?

## Tutorial Extensions

There are many questions about the model performance that can be answered through the limited interface provided by AutoML.

1. How reproducible is the model? If you ran training again with the exact same settings do you get very similar performance metrics and confusion matrices?  
2. We built our model with a 8 hour node budget, but how does it compare to another model with the same underlying data but given longer time to train?  
  **NOTE:** Be aware that the Google AutoML Vision pricing has changed. It will cost ~$20 per wall hour (different from node hour) of model training. (More details here: https://cloud.google.com/automl-tables/pricing). Please avoid training for hundreds of node hours unless it is how to intend to spend your project's financial budget.
4. In our dataset, there is a large imbalance between the classes, which affects the predictions and performance. How might you adjust this imbalance (combining minority classes, upsampling minorities, downsampling majorities)? Modify the 'nih_chest_xray_full.csv' file to implement your model adjustment and train a new model. How is performance affected?
5. Our performance might be elevated because we are seeing multiple images of the same patient in training, validation, and test datasets. Modify the 'nih_chest_xray_full.csv' to carefully control how the images are partitioned across the sets [[formatting guidelines](https://cloud.google.com/vision/automl/object-detection/docs/csv-format)]. One modification that would test the model's generalizability is to train only on females and test and validate only on males (information available in original 'Data_Entry_2017.csv' file at the NIH [Box directory](https://nihcc.app.box.com/v/ChestXray-NIHCC)). Train the new model with the modified data. Does the performance change? What does that say about the generalizability of the previous models?
6. Our model performs multi-label classification with multi-label training examples. But that might not always be the most useful paradigm. Think of a classifier that returns a single one of these labels that might be useful in a clinical setting. How would you modify the dataset for training this model? Train this new model and compare its performance to the original multi-label model. (Remember, if you want to explicitly train a single-label classifier in AutoML, you will need to create a new dataset.)
7. This image data can be used to predict other labels besides disease status. Train a classifier to predict the decade of life for each patient (e.g., a 28 year old is in their twenties). Does this model perform its task better than our model that predicts disease? What privacy implications does this suggest?

### Other Ideas to Explore

In order to further evaluate model performance, you could try training the model with different sized samples of the training data and see how model accuracy changes will less training data. You could also investigate the role of annotation text on the x-ray images plays on the prediction or how robust the model is to image perturbations such as rotations, flips, and zooming. Additionally, you when comparing multiple models, you might want to build a tool that identifies example images with conflicting predictions across models and therefore better understand the limitations or advantages of particular model differences.

## Project Ideas

Once you have completed the tutorial, there are several extensions that could be considered when choosing a project. These might include:

- Improving this model and testing its generalizability with two alternative chest X-ray datasets.
  - Stanford CheXpert Chest X-ray, [dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) and [paper](https://arxiv.org/abs/1901.07031)
  - MIT MIMIC Chest X-ray, [dataset](https://physionet.org/content/mimic-cxr/2.0.0/) and [paper](https://arxiv.org/abs/1901.07042)
- Turning your model into a deployed mobile app **Note: Turning on GCP servers to stand by for uploaded images for this will have daily costs**:
  - AutoML on phone app [instructions](https://cloud.google.com/vision/automl/docs/export-edge)
- Applying AutoML Vision models to different types of images (related [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4977982/))
  - MURA (**mu**sculoskeletal **ra**diographs) [dataset](https://stanfordmlgroup.github.io/competitions/mura/)
  - White blood cells: [tutorial](https://blog.athelas.com/classifying-white-blood-cells-with-convolutional-neural-networks-2ca6da239331), Kaggle [dataset](https://www.kaggle.com/paultimothymooney/blood-cells), Cancer Imaging Archive [dataset](https://www.cancerimagingarchive.net/)
    - Cell Image Library [dataset](http://www.cellimagelibrary.org/browse/organism/Homo%20sapiens)
    - Scoring of Radiographic Joint Damage DREAM [challenge](https://www.synapse.org/#!Synapse:syn20545111/wiki/597243)
- Using alternative vision machine learning software and apply to image datasets with
  - Accompanying genomic or clinical information
  - More than two dimensions ([3D scans](https://www.sciencedirect.com/science/article/pii/S0939388918301181) or videos)
- Rather than classification, find a dataset and method for other image tasks such as image segmentation, de-noising, detection, registration, and synthesis (recent [review](https://www.sciencedirect.com/science/article/pii/S0933365719311510) with accompanying [database](http://livingreview.in.tum.de/GANs_for_Medical_Applications/) of example publications).

## Cleanup

After the tutorial and any project extensions requiring Google Cloud are complete, users will need to delete their Google Cloud projects to halt recurring costs. In the Cloud Console, go to `Manage resources` [page](https://console.cloud.google.com/cloud-resource-manager), find the project you want to delete, click `Delete`, and type the project ID to `Shut down` the project.
