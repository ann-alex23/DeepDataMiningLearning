# DeepDataMiningLearning
Data mining, machine learning, and deep learning sample codes for SJSU CMPE255 Data Mining ([Fall2023 SJSU Official Syllabus](https://sjsu.campusconcourse.com/view_syllabus?course_id=22871&public_mode=1)) and CMPE258 Deep Learning ([Fall2023 SJSU Official Syllabus](https://sjsu.campusconcourse.com/view_syllabus?course_id=26399&public_mode=1)).
* Some google colab examples need SJSU google account to view)
* Large language Models (LLMs) part is newly added
* You can also view the documents in: [readthedocs](https://deepdatamininglearning.readthedocs.io/en/latest/)

## Setups
Install this python package (optional) via

```bash
% python3 -m pip install flit
% flit install --symlink
```
ref "docs/python.rst" for detailed python package description

Open the Jupyter notebook in local machine:
```bash
jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```

## Sphinx docs

Activate python virtual environment, you can use 'sphinx-build' command to build the document

```bash
   % pip install -r requirements.txt
   (mypy310) kaikailiu@kaikais-mbp DeepDataMiningLearning % sphinx-build docs ./docs/build
   #check the integrity of all internal and external links:
   (mypy310) kaikailiu@kaikais-mbp DeepDataMiningLearning % sphinx-build docs -W -b linkcheck -d docs/build/doctrees docs/build/html
```
The generated html files are in the folder of "build". You can also view the documents in: [readthedocs](https://deepdatamininglearning.readthedocs.io/en/latest/)

## Python Data Analytics
Basic python tutorials, numpy, Pandas, data visualization and EDA
* Python tutorial code: [Python_tutorial.ipynb](./Python_tutorial.ipynb)--[colablink](https://colab.research.google.com/drive/1KpLTxgvmFzSlmr486zZwfUBUt-U4-ukT?usp=sharing)
* Python NumPy tutorial code: [Python NumPy tutorial](./Python-Numpy.ipynb)--[colablink](https://colab.research.google.com/drive/10CtxFoyTUk5RIPX4MnOOhYYe3DGAitYW?usp=sharing)
* Data Mining introduction code: 
   * [Dataintro-Pandas.ipynb](./notebooks/Dataintro-Pandas.ipynb) --[colablink](https://colab.research.google.com/drive/14zantNUelF-uPLOXYH8PDzcaPFD-94tc)
   * [Dataintro-EDA.ipynb](./notebooks/Dataintro-EDA.ipynb) --[colablink](https://colab.research.google.com/drive/191Ak-8YzdwJVuCjhFUOJ-WnV0OaDYe2f)
   * [Dataintro-Visualization.ipynb](./notebooks/Dataintro-Visualization.ipynb) --[colablink](https://colab.research.google.com/drive/1zPfz3zma_EriCKvLMShM7jsg5aR_1Cpn)

Python data apps based on streamlit: [streamlittest](dataapps/streamlittest.py)

## Cloud Data Analytics

* Data Mining based on Google Cloud: 
   * Google Cloud access via Colab: [colablink](https://colab.research.google.com/drive/1fmNMY23wzoQQTGoGns1cgTrjOIuj-Qtc?usp=sharing)
      * Configure Gcloud, Google Cloud Storage, Compute Engine, Colab Terminal
   * Google BigQuery with Colab/Jupyter introduction [BigQuery-intro.ipynb](./BigQuery-intro.ipynb) -- [colablink](https://colab.research.google.com/drive/1HREJs7dUZfrJaPP2wApPNtaINpe2Rtey?usp=sharing)
      * Natality dataset and Weather data from Google BigQuery
   * COVID19 Data EDA and Visualization based on Google BigQuery (Fall 2022 updated): [colablink](https://colab.research.google.com/drive/1y4zQl_SxA1DEbjI5XjBuxmXQrx5xI1vE?usp=sharing)
      * COVID NYT data, COVID-19 JHU data
   * Additional Google BigQuery examples: [colablink](https://colab.research.google.com/drive/1eHj3g5qwzp4uhE0j0qagCLj5SBWIbuTL?usp=sharing)
      * Chicago Crime Dataset, Austin Waste Dataset, COVID Racial Dataset (race graph)
   * BigQuery ML examples: [colablink](https://colab.research.google.com/drive/1ZX5X9ryN9fq6R1Sb7kEscPaJjRGx0ft3?usp=sharing)
      * COVID, CREDIT_CARD_FRAUD, Predict penguin weight, Natality, US Census Dataset Classification, time-series forecasting from Google Analytics data

## Machine Learning Algorithm
* Machine Learning introduction: 
   * MLIntro-Regression -- [colablink](https://colab.research.google.com/drive/1atrY6rpfPKs5K1VxddfEOWR5QRarxHiG?usp=sharing)
   * MLIntro-RegressionSKLearn -- [colablink](https://colab.research.google.com/drive/1XUzW9vSqyNM02v9F5ueaLMtFb0VorOA3?usp=sharing)
   * [MLIntro2-classification.ipynb](./MLIntro2-classification.ipynb) --[colablink](https://colab.research.google.com/drive/1znfskFZFo-m7VjnI5vgcxdaPWHvLLG9H?usp=sharing)
      * Breast Cancer Dataset, iris Dataset, BigQuery US Census Income Dataset, multiple classifiers. 
   * DecisionTree -- [colablink](https://colab.research.google.com/drive/15N_qxOY74batHHjTvkh6zoQ0_85bfdDQ?usp=sharing)
      * SKlearn DecisionTree algorithm on Iris dataset, Breast Cancel Dataset, Make moon dataset, and DecisionTreeRegressor. A berif discussion of Gini Impurity.    
   * GradientBoosting -- [colablink](https://colab.research.google.com/drive/1eT68ZVw3F8Dw1ZjYmfPo3wJutS68S80Q?usp=sharing)
      *  Gradient boosting process, Gradient boosting regressor with scikit-learn, Gradient boosting classifier with scikit-learn
   * XGBoost -- [colablink](https://colab.research.google.com/drive/1ZKtpwoRnK8r2fy8ucXoz1K9E98X76dFC?usp=sharing)
      * XGBoost introduction, US Census Income Dataset from Big Query, UCI Dermatology dataset

## Deep Learning
Deep learning notebooks (colab link is better)
* Tensorflow introduction code: [CMPE-Tensorflow1.ipynb](./notebooks/CMPE-Tensorflow1.ipynb) -- [colablink](https://colab.research.google.com/drive/188d4pSon4mSAzhGG54zXjWctTOo7Ds53?usp=sharing)
* Pytorch introduction code: [CMPE-pytorch1.ipynb](./notebooks/CMPE-pytorch1.ipynb) -- [colablink](https://colab.research.google.com/drive/1KZKXqa8FkaJpruUl1XzE7vjvb4pHfMoS?usp=sharing)
* Tensorflow image classification:
   * Road sign data from Kaggle example: [Tensorflow-Roadsignclassification.ipynb](./notebooks/Tensorflow-Roadsignclassification.ipynb), [colablink](https://colab.research.google.com/drive/1W0bwQVXDFakcB7FdQbbkrSdsucNWW7we)
   * Flower dataset example with TF Dataset, TFRecord, Google Cloud Storage, TPU/GPU acceleration: [colablink](https://colab.research.google.com/drive/1_CwebpyvkcTdAW4zbffga6DT58yw0bZO?usp=sharing)
* Pytorch image classification sample: [CMPE-pytorch2.ipynb](./notebooks/CMPE-pytorch2.ipynb), [colablink](https://colab.research.google.com/drive/1PduHOC54R3CpdAl2p_MM1WYzQWof5ovL)

New Deep Learning sample code based on Pytorch (under the folder of "DeepDataMiningLearning")
* Pytorch Single GPU image classification with/without automatic mixed precision (AMP) training: [singleGPU](DeepDataMiningLearning/singleGPU.py)
* Pytorch Multi-GPU DDP test: [testTorchDDP](DeepDataMiningLearning/testTorchDDP.py)
* Pytorch Multi-GPU image classification: [multiGPU](DeepDataMiningLearning/multiGPU.py)
* Pytorch Torchvision image classification (Efficientnet) notebook on HPC: [torchvisionHPC.ipynb](DeepDataMiningLearning/torchvisionHPC.ipynb)
* Pytorch Torchvision vision transformer (ViT) notebook on HPC: [torchvisionvitHPC.ipynb](DeepDataMiningLearning/torchvisionvitHPC.ipynb)
* Pytorch ViT implement from scratch on HPC: [ViTHPC.ipynb](DeepDataMiningLearning/ViTHPC.ipynb)
* Pytorch ImageNet classification example: [imagenet](DeepDataMiningLearning/imagenet.py)
* Pytorch inference example for top-k class: [inference.py](DeepDataMiningLearning/inference.py)
* TIMM models: [testtimm.ipynb](DeepDataMiningLearning/testtimm.ipynb)
* Huggingface Images via Transformers: [huggingfaceimage.ipynb](DeepDataMiningLearning/huggingfaceimage.ipynb)
* Siamese network: [siamese_network](DeepDataMiningLearning/siamese_network.py)
* TensorRT example: [tensorrt.ipynb](DeepDataMiningLearning/tensorrt.ipynb)
* Advanced Image Classification: [githubrepo](https://github.com/lkk688/MultiModalClassifier)
   * General purpose framework for all-in-one image classification for Tensorflow and Pytorch
   * Support for multiple datasets: imagenet_blurred, tiny-imagenet-200, hymenoptera_data, CIFAR10, MNIST, flower_photos
   * Support for multiple custom models ('mlpmodel1', 'lenet', 'alexnet', 'resnetmodel1', 'customresnet', 'vggmodel1', 'vggcustom', 'cnnmodel1'), all models from Torchvision and TorchHub
   * Support HPC training and evaluation
* Object detection (other repo)
   * [MultiModalDetector](https://github.com/lkk688/MultiModalDetector)
   * [myyolov7](https://github.com/lkk688/myyolov7): Add YOLOv5 models with YOLOv7, performed training on COCO and WaymoCOCO dataset.
   * [myyolov5](https://github.com/lkk688/yolov5): My fork of the YOLOv5, convert COCO to YOLO format, changed the code to be the base code for YOLOv4, YOLOv5, and ScaledYOLOv4; performed training on COCO and WaymoCOCO dataset.
   * [WaymoObjectDetection](https://github.com/lkk688/WaymoObjectDetection)
      * Waymo Dataset Conversion to COCO format: WaymoCOCO
      * [torchvision_waymococo_train.py](https://github.com/lkk688/WaymoObjectDetection/blob/master/MyDetector/torchvision_waymococo_train.py): performs Pytorch FasterRCNN training based on converted Waymo COCO format data. This version can be applied for any dataset with COCO format annotation
      * [WaymoCOCODetectron2train.py](https://github.com/lkk688/WaymoObjectDetection/blob/master/2DObject/WaymoCOCODetectron2train.py): WaymoCOCO training based on Detectron2
      * [mymmdetection2dtrain.py](https://github.com/lkk688/WaymoObjectDetection/blob/master/2DObject/mymmdetection2dtrain.py): Object Detection training and evaluation based on MMdetection2D
   * [CustomDetectron2](https://github.com/lkk688/CustomDetectron2)

## Unsupervised Learning
* Unsupervised Learning Jupyter notebooks
  * PCA: [colablink](https://colab.research.google.com/drive/1zho_nKQq8yQ-4IFXxw9GZEcdhVdtOabX?usp=share_link)
    * Numpy/SKlearn SVD, PCA for digits and noise filtering, eigenfaces, PCA vs LDA vs NCA
  * Manifold Learning: [colablink](https://colab.research.google.com/drive/1XkCpm7tsnngB7l7AUcrnIo3rKjyfOZev?usp=share_link)
    * Multidimensional Scaling (MDS), Locally Linear Embedding (LLE), Isomap Embedding, T-distributed Stochastic Neighbor Embedding for HELLO, S-Curve, and Swiss roll dataset; Isomap on Faces; Regression with Mainfold Learning
  * Clustering: [colablink](https://colab.research.google.com/drive/1wOMrFR7AXnSc99mUkpJhMLfshpe5aeGd?usp=share_link) 
    * K-Means, Gaussian Mixture Models, Spectral Clustering, DBSCAN 

## NLP and Text Mining
* Text Mining Jupyter notebooks
   * Text Representations: [colablink](https://colab.research.google.com/drive/1L4gyfPqvqdvWSGy88DXVS-7nta1pGWB8?usp=sharing)
      * One-Hot encoding, Bag-of-Words, TF-IDF, and Word2Vec (based on gensim); Word2Vec WiKi and Shakespeare examples; Gather data from Google and WordCLoud
   * Texrtact and NLTK: [colablink](https://colab.research.google.com/drive/1q6Khw3MGJg2S1q8eOcpgtnbLPS_LD7Uj?usp=share_link)
      * Text Extraction via textract; NLTK text preprocessing
   * Text Mining via Tensorflow-text: [colablink](https://colab.research.google.com/drive/1kcM8zAPWDQa1_82OCl74CZOCgZDofipR?usp=share_link)
      * Using Keras embedding layer; sentiment classification example; prepare positive and negative samples and create a Skip-gram Word2Vec model  
   * Text Classification via Tensorflow: [colablink](https://colab.research.google.com/drive/1NyIjdj4d4lueByK-_17BepKLRXz7oM9e?usp=sharing)
      * RNN, LSTM, Transformer, BERT
   * Twitter NLP all-in-one example: [colablink](https://colab.research.google.com/drive/16Lq8pFyxwIUhFi241FYDrG-VfBBSTsgE?usp=sharing)
      * NTLK, LSTM, Bi-LSTM, GRU, BERT

## Recommendation
* Recommendation
   * Recommendation via Python Surprise and Neural Collaborative Filtering (Tensorflow): [colablink](https://colab.research.google.com/drive/1PNi5Vl4YRCsNdLS-pcODSdgbhBlPUoBI?usp=sharing)
   * Tensorflow Recommender: [colab](https://colab.research.google.com/drive/14tfyPInCyZzcr4sk6zRejHR1847WwVR9?usp=sharing)

## Large Language Models (LLMs) and Apps
Train a basic language modeling task via basic Pytorch and Torchtext WikiText2 dataset in HPC. 
```bash
python nlp/torchtransformer.py

| epoch   1 |  2800/ 2928 batches | lr 5.00 | ms/batch  5.58 | loss  3.31 | ppl    27.49
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 24.00s | valid loss  1.96 | valid ppl     7.08
-----------------------------------------------------------------------------------------
| epoch   2 |   200/ 2928 batches | lr 4.75 | ms/batch  5.84 | loss  3.07 | ppl    21.57
| epoch   2 |  2800/ 2928 batches | lr 4.75 | ms/batch  5.49 | loss  2.58 | ppl    13.26
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 1655.94s | valid loss  1.52 | valid ppl     4.57
-----------------------------------------------------------------------------------------
| epoch   3 |   200/ 2928 batches | lr 4.51 | ms/batch  5.04 | loss  2.41 | ppl    11.15
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 15.41s | valid loss  1.44 | valid ppl     4.22
-----------------------------------------------------------------------------------------
=========================================================================================
| End of training | test loss  1.40 | test ppl     4.06
=========================================================================================
```

Train Masked Language model:
```bash
(mycondapy310) [010796032@cs001 DeepDataMiningLearning]$ python nlp/huggingfaceLM2.py --data_name="eli5" --model_checkpoint="distilroberta-base" --task="CLM" --subset=5000 --traintag="1115CLM" --usehpc=True --gpuid=1 --batch_size=32 --learningrate=2e-5

python nlp/huggingfaceLM2.py
data_type=huggingface data_name=eli5 dataconfig= subset=0 data_path=/data/cmpe249-fa23/Huggingfacecache model_checkpoint=distilroberta-base task=MLM unfreezename= outputdir=./output traintag=1116MLM training=True usehpc=False gpuid=0 total_epochs=8 save_every=2 batch_size=32 learningrate=2e-05
Trainoutput folder: ./output\distilroberta-base\eli5_1116MLM
....
Epoch 1: Perplexity: 12.102828644322578                              | 6590/26360 [16:06:09<34:50:06,  6.34s/it]
 38%|███████████████████████>>> Epoch 2: Perplexity: 15.187707787848385                              | 9885/26360 [24:00:07<28:57:25,  6.33s/it]
 50%|███████████████████████>>> Epoch 3: Perplexity: 15.063201196763071                             | 13180/26360 [31:52:08<23:12:51,  6.34s/it]
 62%|███████████████████████>>> Epoch 4: Perplexity: 16.583895970053355                             | 16475/26360 [39:44:32<17:23:28,  6.33s/it]
 75%|███████████████████████>>> Epoch 5: Perplexity: 16.27479412837067██████▎                       | 19770/26360 [47:36:46<11:34:43,  6.33s/it]
 88%|███████████████████████>>> Epoch 6: Perplexity: 16.424729093343636██████████████████            | 23065/26360 [55:28:38<5:47:18,  6.32s/it]
100%|███████████████████████>>> Epoch 7: Perplexity: 17.22636450834783
```


Train GPT2 language models
```bash
(mycondapy310) [010796032@cs001 DeepDataMiningLearning]$ python nlp/huggingfaceLM2.py --model_checkpoint="gpt2" --task="CLM" --traintag="1115gpt2" --usehpc=True --gpuid=2 --batch_size=16
```
Train llama2 7b model and only unfreeze the last layers "model.layers.31" (need 500GB) or "lm_head" (need 40GB)
```bash
(mycondapy310) [010796032@cs001 DeepDataMiningLearning]$ python nlp/huggingfaceLM2.py --model_checkpoint="Llama-2-7b-chat-hf" --task="CLM" --unfreezename="lm_head" --traintag="1115llama2" --usehpc=True --gpuid=2 --batch_size=8

python nlp/huggingfaceLM2.py --model_checkpoint="Llama-2-7b-chat-hf" --pretrained=="/data/cmpe249-fa23/trainoutput/huggingface/Llama-2-7b-chat-hf/eli5_1115llama2/savedmodel.pth" --task="CLM" --unfreezename="lm_head" --traintag="1119llama2" --usehpc=True --gpuid=0 --batch_size=8
.....
Epoch 0: Perplexity: 9.858825392857694████████████████████████████████████████| 2627/2627 [12:39:17<00:00,  3.30s/it]
 25%|█████████████████▊     >>> Epoch 1: Perplexity: 10.051054027867561     | 21014/84056 [22:50:31<56:09:49,  3.21s/it]
 38%|███████████████████████>>> Epoch 2: Perplexity: 10.181400762228291     | 31521/84056
```

Train translation models based on huggingfaceSequence
```bash
python nlp/huggingfaceSequence.py --data_name="kde4" --model_checkpoint="Helsinki-NLP/opus-mt-en-fr" --task="Seq2SeqLM" --traintag="1116" --usehpc=True --gpuid=0 --batch_size=8

epoch 0, BLEU score: 51.78█████████████████████████████████████████████████████████████████████████████████████████████████| 2628/2628 [21:18<00:00,  3.00it/s] 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2628/2628 [22:54<00:00,  1.91it/s]
epoch 1, BLEU score: 52.73█████████████████████████████████████████████████████████████████████████████████████████████████| 2628/2628 [22:54<00:00,  2.81it/s] 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2628/2628 [23:06<00:00,  1.90it/s]
epoch 2, BLEU score: 54.
epoch 3, BLEU score: 54.
epoch 4, BLEU score: 55.
epoch 5, BLEU score: 55.
epoch 6, BLEU score: 54.
epoch 7, BLEU score: 55.
```


NLP models based on Huggingface Transformer libraries
* Starting
   * [HuggingfaceTransformers](notebooks/Transformers.ipynb)
   * [huggingfacetest](nlp/huggingfacetest.py)
   * [hfdataset.py](nlp/hfdataset.py)
   * [huggingfaceHPC.ipynb](nlp/huggingfaceHPC.ipynb)
   * [huggingfaceHPCdata](nlp/huggingfaceHPCdata.py)
* Classification application
   * [BERTMTLfakehate](nlp/BERTMTLfakehate.py)
   * [MLTclassifier](nlp/MLTclassifier.py)
   * [huggingfaceClassifierNER.ipynb](nlp/huggingfaceClassifierNER.ipynb)
* Multi-modal Classifier: [huggingfaceclassifier2](nlp/huggingfaceclassifier2.py), [huggingfaceclassifier](nlp/huggingfaceclassifier.py)
* Sequence related application, e.g., translation, summary
   * [huggingfaceSequence](nlp/huggingfaceSequence.ipynb)
* Question and Answer (Q&A)
   * [huggingfaceQA.py](nlp/huggingfaceQA.py)
* Chatbot
   * [huggingfacechatbot.ipynb](nlp/huggingfacechatbot.ipynb)

Pytorch Transformer
* [torchtransformer](nlp/torchtransformer.py)

Open Source LLMs
* [BERTLM.ipynb](nlp/BERTLM.ipynb)
* Masked Language Modeling: [huggingfaceLM.ipynb](nlp/huggingfaceLM.ipynb)
* [llama2](nlp/llama2.ipynb)

LLMs Apps based on OpenAI API
* [openaiqa.ipynb](dataapps/openaiqa.ipynb), [webcrawl.ipynb](dataapps/webcrawl.ipynb)

LLMs Apps based on LangChain
* [langchaintest.ipynb](nlp/langchaintest.ipynb)

