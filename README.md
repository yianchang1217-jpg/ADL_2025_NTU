How to inference:
•	Run download.sh
•	cd to ADL2025/Infer
•	Inference code are in Infer/infer_mc.py and  Infer /infer_qa.py
Then run bash ./run.sh ./data/context.json ./data/test.json ./prediction.csv and replace the paths as you wish
•	The dataset should store in /data.


How to train:
•	Training code are in Train/ run_swag_no_trainer.py and Train/ run_qa_no_trainer.py
•	Run bash train.sh to train models, please change the output_dir before you run it, also make sure the data are stored in the correct directory.
•	Adjust the hyperparameters by yourself.
•	For plotting , code in Q3.
•	For training a Not Pre-trained model , code in Q4.
•	For training an end to end model , code in Q5 /run_longformer_qa_allinone.py .Run bash train_longformer.sh


```python

```
