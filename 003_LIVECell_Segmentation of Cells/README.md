# Segmentation of Cells in Phase-Contrast Microscopy Images with the U-Net Convolutional Neural Network 
---

## Dataset Used

**LIVECell Dataset**  
The dataset contains high-quality annotated cell images collected using the Incucyte live-cell imaging system.

**Link to Dataset:**  
[https://github.com/sartorius-research/LIVECell](https://github.com/sartorius-research/LIVECell)

**Dataset Details**:
- A total of **3,727 images** from multiple cell lines at different confluencies were used.
- Of these, **3,188 images** had matched annotations provided in the `LIVECell_train.json` file.

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

**Status**: _In Progress_  
The model is currently being trained. Results will be updated once training is complete.
