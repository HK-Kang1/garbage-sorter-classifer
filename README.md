### Multimodal Garbage Classification with Image and Text

This project implements a multimodal deep learning classifier that combines visual and textual data to classify garbage into four categories. The model integrates **MobileNetV2** for image feature extraction and **DistilBERT** for text embeddings extracted from image filenames, which often provide contextual information about the garbage type.

Key components include:
- **Multimodal Fusion**: Combines image and text features using a custom PyTorch neural network.
- **Preprocessing**: Uses `torchvision` for image normalization and `transformers` for text tokenization.
- **Custom Dataset**: Loads image, text, and label with tokenization and transformations.
- **Training Pipeline**: Includes training, validation, and testing phases with performance tracking.
- **Performance**: Achieves ~90.6% validation accuracy and ~85.9% test accuracy after 5 epochs.
- **Evaluation**: Final predictions are analyzed with a confusion matrix for class-wise insights.

🔧 **Tech Stack**: PyTorch, torchvision, Hugging Face Transformers, MobileNetV2, DistilBERT, matplotlib, seaborn.
