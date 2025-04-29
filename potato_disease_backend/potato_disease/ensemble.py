import numpy as np

# Classes
classes = ['Early Blight', 'Late Blight', 'Healthy']

def soft_voting(preds1, preds2, preds3, preds4):
    """ Perform Soft Voting: average the predictions """
    return (preds1 + preds2 + preds3 + preds4) / 4

def weighted_voting(preds1, preds2, preds3, preds4, weights=[0.35, 0.35, 0.15, 0.15]):
    """ Perform Weighted Voting: apply custom weights to model predictions """
    return preds1 * weights[0] + preds2 * weights[1] + preds3 * weights[2] + preds4 * weights[3]

def get_final_prediction(avg_preds):
    """ Return the final predicted class based on the averaged probabilities """
    if avg_preds is None or len(avg_preds) == 0:
        print("Error: Invalid predictions received")
        return None  # Return None if predictions are invalid

    final_index = np.argmax(avg_preds)
    confidence = round(avg_preds[final_index] * 100, 2)
    print("Final Prediction class:", classes[final_index])  # Log the class
    print("Confidence:", confidence)  # Log the confidence

    return {
        'class': classes[final_index],
        'confidence': float(confidence)  # Convert confidence to native Python float
    }

