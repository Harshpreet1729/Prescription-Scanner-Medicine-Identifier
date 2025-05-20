PRESCRIPTION SCANNER & MEDICINE IDENTIFIER
==========================================

A deep learning-based system that interprets handwritten medical prescriptions
to identify medicine names. The model is built using a modified ResNet18 CNN 
architecture to classify prescription images and map the identified medicine 
names to their respective medical purposes.

⚠️ Note: This project is experimental and not production-ready. Predictions may
be inaccurate, especially for prescriptions with unclear or highly stylized handwriting.

--------------------------------------------------
FEATURES
--------------------------------------------------
- Accepts grayscale prescription images with augmentation (rotation, affine, perspective)
- Custom ResNet18 CNN trained for handwritten word classification
- Maps predictions to medical purposes using an external CSV
- ~88% validation accuracy on the current dataset
- Visual metrics: training loss plots, precision/recall reports

--------------------------------------------------
MODEL DETAILS
--------------------------------------------------
- Architecture: Modified ResNet18
- Input: 64x64 grayscale images
- Loss Function: CrossEntropyLoss (with class weights)
- Optimizer: Adam (lr = 0.0005)
- Scheduler: StepLR (halves learning rate every 5 epochs)
- Epochs: 15
- Batch Size: 32

--------------------------------------------------
RESULTS
--------------------------------------------------
- Achieved ~88% validation accuracy
- Classification report shows good precision/recall across most classes
- Trained model saved as 'ocr_cnn_model.pth'
- Label encoding stored in 'label_encoder.pkl'

--------------------------------------------------
LIMITATIONS
--------------------------------------------------
- Small and imbalanced dataset limits performance
- Difficulty with medicine names that look similar in handwriting
- High variation in handwriting styles affects accuracy
- Not suitable for real-world deployment yet

--------------------------------------------------
FUTURE WORK
--------------------------------------------------
- Train on a larger and more diverse dataset
- Add OCR capabilities to process full prescription sheets
- Implement spelling correction and synonym matching
- Deploy as a mobile or web app for real-time usage


--------------------------------------------------
USAGE
--------------------------------------------------
To train the model:
   python train_model.py

To predict a medicine from an image:
   python predict_from_image.py --image path_to_image.jpg

Outputs:
- Predicted medicine name
- Associated purpose from Medicine_Details.csv

# 📁 Folder Structure

```plaintext
AI_PROJECT_ROOT/
│
├── ai/
├── data_to_test/
├── dataset/
│   ├── Testing/
│   │   ├── testing_words/
│   │   └── testing_labels.csv
│   ├── Training/
│   │   ├── training_words/
│   │   └── training_labels.csv
│   ├── Validation/
│   │   ├── validation_words/
│   │   └── validation_labels.csv
│   └── Medicine_Details.csv
│
├── gui_app.py
├── main.py
├── model.py
├── predict.py
├── train_model.py
├── utils.py
├── label_encoder.pkl
├── ocr_cnn_model.pth
├── ocr_cnn_model_best.pth
├── training_loss_plot.png
├── confusion_matrix.png
└── README.md
