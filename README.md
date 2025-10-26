Frog Species Identification Using CNN (MobileNetV2)
Problem Statement

Accurate identification of frog species is critical for biodiversity studies and conservation efforts. Traditional identification methods are labor-intensive and require expert taxonomic knowledge. This project aims to automate frog species identification in India by combining image classification with descriptive feature matching, using a Convolutional Neural Network (CNN) based on MobileNetV2.

Dataset

Sources:

iNaturalist (via pyinaturalist API)

AmphibiaWeb

Pexels API (for species with no available images)

Content:

Images of 210 Indian frog species

Manually curated and organized into folders named after each species

Data augmentation applied (flipping, rotation, noise addition, brightness adjustment) to increase dataset diversity

Data Format:

Training and validation images stored in separate directories

Each species folder contains its respective images

Model Architecture

Base Model: MobileNetV2 (pretrained on ImageNet, include_top=False)

Chosen for efficiency and accuracy, suitable for real-time applications

Custom Layers:

GlobalAveragePooling2D

Dense layer with 1024 units, ReLU activation

Output layer with softmax activation for classification

Transfer Learning:

Base layers frozen during initial training

Fine-tuned later with a reduced learning rate to improve species-specific accuracy

Optimizer: Adam

Loss Function: Categorical Crossentropy

Input Image Size: 224x224x3

Preprocessing

Image Resizing: All images resized to 224x224

Normalization: Pixel values scaled to [0,1]

Augmentation:

Horizontal and vertical flips

Rotation (-20° to 20°)

Additive Gaussian noise

Brightness adjustments

Implementation

Framework: TensorFlow + Keras

Key Steps:

Load and preprocess dataset

Define MobileNetV2-based CNN with custom classifier

Train the model using augmented training data

Fine-tune the model on all layers

Save the trained model for deployment

Implement a user interface for image and feature input

Combine image classification with fuzzy string matching for feature-based identification

Display results with frog images and species information

Feature Matching:

User-provided descriptors (type locality, coloration, leg length, skin type, elevation) matched to dataset using fuzzy string matching (fuzzywuzzy)

Decision logic prioritizes feature-based match if model confidence is below 0.75 or results mismatch

Instructions for Running the Code

Place your frog images in the frog_images_per_head folder, organized by species.

Ensure the Excel file CpiFinal.xlsx with species information is in the working directory.

Load the pretrained model frog_species_classifier1.keras.

Run the main script:

python frog_species_identifier.py


Provide the image filename and feature descriptions when prompted.

The program displays the matched frog species image and detailed information.

Evaluation Metrics and Results

Accuracy: High accuracy achieved with MobileNetV2 on augmented dataset

Decision Logic: Combines CNN confidence with fuzzy feature matching to improve identification reliability

Visualization: Predicted species displayed with its image and all 22 dataset attributes

Insights & Challenges

Challenges:

Limited availability of images for certain species

Need for robust augmentation to handle intra-species variability

Aligning image-based and feature-based identification results

Solutions:

Used Pexels API for species with no images

Applied multiple augmentation techniques

Implemented fuzzy string matching to handle partial or approximate feature inputs

Outcome:

Automated identification system capable of recognizing 210 Indian frog species

Supports conservation studies and real-time field applications

References

MobileNetV2: Howard et al., 2017

iNaturalist API: pyinaturalist Python package

AmphibiaWeb: https://amphibiaweb.org

Pexels API for image retrieval
