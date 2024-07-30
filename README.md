# deep-learning-challenge
Module 21 challenge

**Report on the Performance of the Deep Learning Model for Alphabet Soup**

**Overview of the Analysis**

The primary objective of this analysis was to develop and refine a deep learning model that can predict the likelihood of success for organizations funded by Alphabet Soup. This involved several key steps: preprocessing the dataset to make it suitable for model training, designing a neural network with appropriate architecture, training the model, and optimizing it to achieve an accuracy higher than 75%. The ultimate goal was to provide Alphabet Soup with a tool that can accurately assess which funding applications are most likely to be successful based on historical data.

To achieve this, the following tasks were carried out:

1. **Data Preprocessing:**
   - **Cleaning and Transformation:** Removed non-informative columns and encoded categorical variables.
   - **Feature Engineering:** Managed categorical variables and normalized features.
   
2. **Model Design and Training:**
   - **Neural Network Architecture:** Designed a neural network with multiple hidden layers and neurons to capture complex patterns in the data.
   - **Training:** Used 200 epochs to ensure thorough learning.
   - **Evaluation:** Assessed the model’s performance using accuracy as the primary metric.

3. **Optimization:**
   - **Model Refinement:** Adjusted the network architecture and training parameters to improve performance.
   - **Performance Monitoring:** Evaluated the model's accuracy and made iterative adjustments.

**Results**

**Data Preprocessing**

- **Target Variable:**
  - `IS_SUCCESSFUL`: This binary outcome variable indicates whether the funding was used effectively by the organizations.

- **Feature Variables:**
  - Variables used as features in the model included:
    - `APPLICATION_TYPE`
    - `AFFILIATION`
    - `CLASSIFICATION`
    - `USE_CASE`
    - `ORGANIZATION`
    - `STATUS`
    - `INCOME_AMT`
    - `SPECIAL_CONSIDERATIONS`
    - `ASK_AMT`

- **Variables Removed:**
  - `EIN`: An identification column that did not contribute to the predictive power of the model.
  - `NAME`: An identification column that was deemed irrelevant for the prediction task.

**Compiling, Training, and Evaluating the Model**
Several models were tried with different characteristics but all did not yeild an accuracy greater than 75%. See below:
![Screenshot 2024-07-29 212355](https://github.com/user-attachments/assets/bbed7958-b39f-4b6e-b720-2d26f92e824c)

In this model,4 layers were used with the first layer having 100 neurons and these gradually reduced. However, the accuracy did not improve from the initial attempt.


![Screenshot 2024-07-29 212259](https://github.com/user-attachments/assets/9f35e460-dd2f-4651-9890-8f5ca7b98036)

In the above model, 4 layers were used also but the first layer was set to 150 neurons and these were gradually reduced. However, the accuracy did not improve beyond 75%.



The final model used has the characteristics below:

- **Neural Network Architecture:**
  - **Input Layer:**
    - The number of neurons matched the number of features after preprocessing and encoding.
  - **Hidden Layers:**
    - **First Hidden Layer:** 200 neurons with ReLU activation function, chosen for its ability to model complex non-linear relationships.
    - **Second Hidden Layer:** 180 neurons with ReLU activation function, to capture intermediate-level features.
    - **Third Hidden Layer:** 140 neurons with ReLU activation function, to further abstract the features.
    - **Fourth Hidden Layer:** 100 neurons with ReLU activation function, focusing on high-level abstractions.
    - **Fifth Hidden Layer:** 50 neurons with ReLU activation function, to refine the feature representations before the output layer.
  - **Output Layer:**
    - 1 neuron with sigmoid activation function, providing a probability score for binary classification.

- **Performance:**
  - Despite the sophisticated architecture and extended training duration, the model’s accuracy remained at approximately 75%, which did not surpass the initial performance. Various optimization attempts, including adjusting neuron counts and training epochs, did not lead to significant improvements. The absence of early stopping and callbacks likely affected the model's ability to optimize effectively.

- **Steps Taken to Improve Model Performance:**
  - **Feature Engineering:**
    - Combined rare categorical values into an "Other" category to manage the dimensionality and improve model performance.
    - Applied `StandardScaler` to normalize feature values, ensuring consistent scale and aiding convergence.
  - **Model Adjustments:**
    - Implemented a complex architecture with 5 hidden layers and varying numbers of neurons to capture intricate patterns.
    - Increased the number of epochs to 200 to allow the model sufficient time for learning.

**Summary**

The deep learning model, with its extensive architecture and training adjustments, achieved an accuracy of 72.5%, lower than the initial result. The lack of early stopping and callbacks potentially limited the model's effectiveness by not preventing overfitting or saving the best-performing version.This model may thus not be very reliable in making predictions of success of organisations funded by the charity as a 25% chance of error is significant.

**Recommendations:**
- **Explore Alternative Models:** Consider using other machine learning algorithms like Random Forests or Gradient Boosting Machines, which might offer better performance or require less extensive parameter tuning.
- **Implement Early Stopping and Callbacks:** Utilize these techniques to monitor training, prevent overfitting, and save the best model, which could enhance performance.
- **Expand Hyperparameter Tuning:** Perform a comprehensive hyperparameter search to optimize settings such as learning rates and batch sizes for better model performance.

In conclusion, while the neural network model was well-designed and thoroughly optimized, it did not achieve the desired performance improvements. Further exploration of different models and additional optimization techniques could help achieve better accuracy and more reliable predictions for Alphabet Soup.
