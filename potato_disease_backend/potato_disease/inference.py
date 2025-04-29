import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from potato_disease.ensemble import soft_voting, weighted_voting

# Load models
model1 = load_model('potato_disease/models/model1.h5', compile=False)
model2 = load_model('potato_disease/models/model2.h5', compile=False)
model3 = load_model('potato_disease/models/model3.h5', compile=False)
model4 = load_model('potato_disease/models/model4.h5', compile=False)

# Classes
classes = ['Early Blight', 'Late Blight', 'Healthy']

def preprocess_image(img):
    img = img.resize((256, 256))  # Ensure image is resized to (256, 256)
    img_array = image.img_to_array(img)   # Normalize pixel values
    img_tensor = np.expand_dims(img_array, axis=0)  # Add batch dimension
    print("Image tensor shape:", img_tensor.shape)  # Log the shape
    return img_tensor

def predict_ensemble(img):
    img_tensor = preprocess_image(img)

    # Get predictions from each model
    preds = [model.predict(img_tensor)[0] for model in [model1, model2, model3, model4]]
    
    for i, pred in enumerate(preds, start=1):
        percentage = [f"{x * 100:.2f}%" for x in pred]
        print(f"Model {i} Prediction: {percentage}")

    # Get both voting results
    soft_pred = soft_voting(*preds)
    weighted_pred = weighted_voting(*preds)

     # Print final voting arrays as percentages
    print("Soft Voting Final:", [f"{x * 100:.2f}%" for x in soft_pred])
    print("Weighted Voting Final:", [f"{x * 100:.2f}%" for x in weighted_pred])

    def format_result(final_pred):
        idx = np.argmax(final_pred)
        raw_conf = float(final_pred[idx])            # raw probability
        disp_conf = round(raw_conf * 100, 2)         # percentage display
        preds_list = [round(float(x), 4) for x in final_pred]
        return {
            'class': classes[idx],
            'confidence': disp_conf,       # percentage
            'confidence_display': disp_conf,  # Ensure confidence_display is returned
            'predictions': preds_list
        }

    return {
        'soft': format_result(soft_pred),
        'weighted': format_result(weighted_pred)
    }



# Test function (this is just for testing the model)
def test_model():
    test_img = Image.open("C:/Users/subha/Potatodisease/PlantVillage/Potato___healthy/3a1dbeee-089c-43f0-8f51-a92d3687a515___RS_HL 1754.JPG")  # Replace with a test image path
    test_img = preprocess_image(test_img)  # Preprocess the image
    prediction = model1.predict(test_img)  # Test one model
    print("Test model prediction:", prediction)

    # Convert prediction (numpy float32 array) to list of floats
    prediction_list = prediction.astype(float).tolist()

    return prediction_list
