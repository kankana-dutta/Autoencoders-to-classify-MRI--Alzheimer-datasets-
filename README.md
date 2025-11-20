****Semi-Supervised Autoencoder for Alzheimer’s MRI Classification****<br>

This project implements a semi-supervised convolutional autoencoder that simultaneously:<br>
   a. Reconstructs MRI slices<br>
   b.Classifies dementia stages<br>
   c.Compare latent dimension between demented and non-demeneted groups<br>
   
**The model is trained on the Falah/Alzheimer_MRI dataset from HuggingFace and predicts one of four classes**:<br>
  0-Non demented<br>
  1-Very mild demented<br>
  2-Mild demented<br>
  3-Moderate Demented<br>

**Model Architecture**<br>
  Input MRI (128×128) <br>
        │
   [ Convolutional Encoder ] <br>
        │
   Latent Vector (64-D) <br>
        ├───────────────► [ Fully-Connected Classifier → Dementia Stage ] <br>
        ▼
   [ Transpose-Conv Decoder ] <br>
        ▼
     Reconstructed MRI <br>
     
**Attachments**<br>

<img width="1000" height="800" alt="Reconstruted_MRI_images" src="https://github.com/user-attachments/assets/4bc19a07-6b01-47c2-8a20-745a767d5a3f" /> <br>

<img width="700" height="400" alt="Training_curves" src="https://github.com/user-attachments/assets/6d5e8e34-11c0-4aff-9b6c-0e77025e9266" /><br>

<img width="1000" height="500" alt="Latent space variation" src="https://github.com/user-attachments/assets/0401fcfa-1259-4082-a149-a632eb326041" /><br>


**Results**<br>
The semi-supervised autoencoder successfully learned a compact and meaningful representation of MRI anatomy while simultaneously predicting dementia stage. The decoder reconstructs anatomically plausible brain slices across all four clinical categories, capturing major structural elements, and global tissue contrast. Latent-space analysis the mean ±1σ values for non-demented and demented groups show only modest separation across most of the 64 latent dimensions. Low variance across dimensions suggests that the model has learned a relatively smooth, compact latent manifold where both healthy and demented MRIs occupy overlapping regions with subtle statistical differences rather than large, well-separated clusters. In other words, dementia-related variations—such as hippocampal shrinkage or ventricular expansion—are present in the latent code, but they do not dominate the representation strongly enough to create clear, axis-aligned separations.Training curves demonstrate rapid convergence of both reconstruction and classification losses, stabilizing within the first ~10 epochs, indicating efficient optimization of both supervised and unsupervised objectives.<br>

