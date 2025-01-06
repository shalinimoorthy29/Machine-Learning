# Cell Segmentation in Phase-Contrast Microscopy Using Deep Learning

## Dataset Used

**LIVECell Dataset**  
The dataset contains high-quality annotated cell images collected using the Incucyte live-cell imaging system.

**Link to Dataset:**  
[https://github.com/sartorius-research/LIVECell](https://github.com/sartorius-research/LIVECell)

**Dataset Details**:
- A total of **679 images** from multiple cell lines at different confluencies were used.
- Of these, **572 images** had matched annotations provided in the `LIVECell_train.json` file.

## Objectives

- To develop a deep learning model for **semantic segmentation** of cells in phase-contrast microscopy images.
- To evaluate cell confluence rather than direct cell count.
- To experiment with and compare different segmentation methods.

## Methods

1. **Data Preprocessing**:
   - Images and masks were preprocessed into a standardised shape of **512x512 pixels**.
   - Masks were validated and adjusted to ensure consistent dimensions.

2. **Model Architecture**:
   - A simplified U-Net architecture was implemented with the following:
     - **Encoder**: Three downsampling layers using convolution and max-pooling.
     - **Bottleneck**: Convolution layers to capture deep feature representations.
     - **Decoder**: Three upsampling layers with skip connections to restore spatial resolution.

3. **Segmentation Methods**:
   - The **watershed algorithm** was attempted for **instance segmentation** but requires further refinement for improved results.
   - For now, focusing on **semantic segmentation** to predict confluence instead of direct cell count.

4. **Training Configuration**:
   - **Loss Function**: Binary cross-entropy.
   - **Metrics**: Accuracy, IoU, and Dice coefficient.
   - **Optimiser**: Adam optimiser with a learning rate of `1e-4`.
   - **Batch Size**: 16.
   - **Epochs**: 20.
   - **Callbacks**: Early stopping and learning rate reduction on plateau were used to prevent overfitting.

5. **Evaluation**:
   - Training and validation sets were split in an **80-20** ratio.
   - Results were evaluated based on segmentation accuracy and overlap metrics.

## Results and Conclusion

### Summary of Training Output

**Dataset Overview:**

- Training Data: 457 images, each of size 512 x 512 with 3 channels.
- Validation Data: 115 images, also of size 512 x 512 x 3.
- Masks: Mismatched dimensions in masks were addressed during preprocessing to ensure compatibility with the model.

**Model Performance:**

Final Training Metrics:
- Accuracy: 85.82%
- Dice Coefficient: 0.7987
- IoU (Intersection over Union): 0.6659
- Loss: 0.3152

Final Validation Metrics:
- Accuracy: 86.01%
- Dice Coefficient: 0.7865
- IoU: 0.6493
- Loss: 0.3065

**Key Observations:**
- The model showed steady improvements in metrics, with significant increases in Dice coefficient and IoU between epochs 5–10.
- Early stopping did not trigger, indicating that the model was still improving at the maximum configured epochs.
- The validation Dice coefficient of **0.7865** and IoU of **0.6493** indicate good segmentation performance for the dataset size.

## Next Steps

To improve the segmentation performance, particularly the Dice coefficient, IoU, and accuracy, consider the following tuning methods:

1. **Increase Epochs**:
   - Extend the training to **30 epochs** or more, allowing the model to learn further patterns.
   - Early stopping will ensure training halts if overfitting occurs.

2. **Data Augmentation**:
   - Introduce techniques such as flipping, rotation, scaling, and brightness adjustments to artificially expand the dataset and improve model generalisation.

3. **Learning Rate Scheduling**:
   - Experiment with lower initial learning rates (e.g., `1e-5`) or use learning rate warm-up strategies for smoother convergence.

4. **Batch Size Optimisation**:
   - Test with larger batch sizes (e.g., 32 or 64) to stabilise gradient updates, provided memory resources allow.

5. **Model Architecture**:
   - Replace the U-Net's encoder with a **pre-trained backbone** such as ResNet or EfficientNet to leverage transfer learning.
   - Add more layers or increase filter sizes in the encoder and decoder for better feature extraction.

6. **Loss Function**:
   - Experiment with **Dice loss** or **Tversky loss**, which are tailored for imbalanced segmentation tasks.

7. **Hyperparameter Tuning**:
   - Use tools like **Optuna** or **GridSearchCV** to systematically explore combinations of hyperparameters (e.g., learning rate, dropout, regularisation strength).

8. **Post-Processing**:
   - Refine predictions with techniques such as **CRF (Conditional Random Fields)** or morphological operations to improve mask quality.

## Visualisation

- Include plots of the learning curves for training and validation accuracy, Dice coefficient, IoU, and loss.
- Visualise sample segmentation results (ground truth vs predicted masks) to assess qualitative performance.

## Acknowledgements

Special thanks to the developers of the **LIVECell Dataset** and the community contributing to advancements in deep learning-based microscopy analysis.
