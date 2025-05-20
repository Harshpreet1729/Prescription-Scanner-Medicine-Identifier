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
