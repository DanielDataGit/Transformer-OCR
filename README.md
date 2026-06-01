
# README
## 1. Project Overview

Project Title: Transformer OCR

Model Type:
Seq2Seq Transformer

Objective:
Generative

Dataset Used:
Personally collected dataset + https://www.kaggle.com/datasets/danieltafmizi/ocr-label-crops 

Expected test evaluation for sanity check: 
Daniels proposed model) 68% Word Accuracy and 80% character accuracy
Alexs foundation) 94% word accuracy and 98% character accuracy
Alexs proposed)  0.009346% word accuracy and 0.239344% character accuracy
Savithas proposed model) did not provide a testing method
------------------------------------------------------------

## 2. Repository Structure


```
project_root/

  Daniel/
  	baselineEvaluation/
  		baselineModel.ipynb (Baseline model script)
  		eval.ipynb (Evaluation metric class and baseline evaluation results
  	dataPreparation/
  		dataPrep.ipynb (script for extracting crops and spltting)
  		labelStudio.ipynb (Builds json of crops for manual annotation using label studio)
  	final_model_scripts/
  		experimental.ipynb (experimentation and development)
  		final_model_inference.py (runs testing)
  		final_model_train.py (runs training)
  	Model_weights/
  		final_model_weights/ (folder with final weights from proposed model)
  		stage1_sroie_weights/ (folder with weights from sroie pretraining)
  		customYOLO26N.pt (weights for custom yolo object detection model)
  	DeepLearningP2P2-personal.doxc (project report)
  	
  Savitha/
  	test/
  		rawCropTest
  		tradOutputTest
  	train/
  		rawCropTrain
  		tradOutputTrain
  	val
  		rawCropVal
  		tradOutputVal
  	training_graphs_Savitha Namelikonda/ (folder with training result graphs)
  	.py files (her proposed model with different improvement techniques)
  	.png files (The output of her model with the different improvement techniques)
  	.pdf file (project report)
  	
  Alexander/
  	Project 2/
  		Foundation Model scripts/ (train and inference for foundational model)
  			inference.py
  			training.py
  		Model Weights/ 
  			foundation_model_weights/
  			improved_trocr/
  		proposed model scripts/ (train and inference for proposed model)
  			inference.py
  			training.py
  		Project 2 report.docx
  		
  data/
  	OCRcropsv3/
  	test/
  		rawCropTest
  		tradOutputTest
  	train/
  		rawCropTrain
  		tradOutputTrain
  	val
  		rawCropVal
  		tradOutputVal
  	
  README_project2.txt
  
  daniel_requirements.txt
  savitha_requirements.txt
  Deep Learning Project 2 Final Presentation.mp4
```

------------------------------------------------------------

## 3. Dataset 

Box Link to Dataset:

https://usf.box.com/s/zzks93cmxk9oc06k8tr2q80gffwekq6j


Savitha changed the csv files in my dataset, and thus her scripts do not work with the dataset above. Below are her datasets, I apologize. Also, she was passing absolute paths for the images, so I had to fix that.

https://usf.box.com/s/wluey75zim5pqz4uid9au6na5cir45ub
https://usf.box.com/s/tx95cs5sacq9z6b6f2ywxqgfh6jx9za3
https://usf.box.com/s/40m26cg5iwl889flie4r3f8ig8dpirug


Give access to the following emails:
yusun@usf.edu, kandiyana@usf.edu

Where to place the dataset after download:
```

Daniels dataset goes)

data/
	OCRcropsv3/
		data goes here
		
		
		
Savithas dataset goes)

Savitha/
  	test/
  		rawCropTest
  		tradOutputTest
  	train/
  		rawCropTrain
  		tradOutputTrain
  	val
  		rawCropVal
  		tradOutputVal
  		
```

------------------------------------------------------------

## 4. Model Checkpoint

Box Link to Best Model Checkpoint:

Daniels weights)https://usf.box.com/s/zwyfgd7vkpwhqjiho873fb6gvigwxbly

Savithas weights) Savitha did not provide any weights

Alexanders weights) https://usf.box.com/s/p39n560tt5sea1xi4fc69eywejdc29ws

Where to place the checkpoint after downloading:
```
Daniels weights)
  	Model_weights/
  		final_model_weights/ (folder with final weights from proposed model)
  		stage1_sroie_weights/ (folder with weights from sroie pretraining)
  		customYOLO26N.pt (weights for custom yolo object detection model)

Alexanders weights)
		Model Weights/ 
  			foundation_model_weights/
  			improved_trocr/

```

------------------------------------------------------------

## 5. Requirements (Dependencies)

Python Version:
Daniels version) 3.12.3, developed in linux
Alexanders Version) 3.11
Savithas version) 3.13.2

How to install all dependencies (e.g. requirements.txt):

install Daniels dependencies) pip install -r daniel_requirements.txt
install Alexanders dependencies) pip install -r alexander_requirements.txt
install Savithas dependencies) pip install -r savitha_requirements.txt

Using pip:
```
pip install -r requirements.txt
```

Using conda (creates env and installs):
```
conda create -n dlproj python={{your_python_version}}
conda activate dlproj
pip install -r requirements.txt
```

------------------------------------------------------------

## 6. Running the Test Script

Command to run testing:

Daniels testing) python "Daniel/final_model_scripts/final_model_inference.py"

Alexs foundation test) python "Alexander/Project 2/Foundation Model Scripts/inference.py" 

Alexs proposed test) python "Alexander/Project 2/Proposed Model Scripts/inference.py"

Savithas testing) Savitha did not provide a testing script

------------------------------------------------------------

## 7. Running the Training Script

Command to run training:

Daniels training) python "Daniel/final_model_scripts/final_model_train.py"

Alexs foundation training) python "Alexander/Project 2/Foundation Model Scripts/inference.py"

Alexs proposed training) python "Alexander/Project 2/Proposed Model Scripts/training.py"

Savithas training) python "Savitha/Autoregressive_TrOCR_model - withepoch200_dropout0.2_withplots_Savitha Namelikonda.py"

------------------------------------------------------------


