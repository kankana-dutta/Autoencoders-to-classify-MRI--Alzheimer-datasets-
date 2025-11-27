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

**CNN and Autoencoder Perform Equally**<br>
Both the CNN and the semi-supervised autoencoder achieve nearly identical classification accuracy in identifying Alzheimer’s patients from MRI data, despite using very different learning strategies. This equivalence in performance makes the autoencoder particularly interesting, because it compresses each brain slice into a compact 64-dimensional latent space while still preserving all disease-relevant information needed for accurate diagnosis. Since the latent representation is low-dimensional, continuous, and model-generated, it opens up the possibility of exploring how entire patient populations map into this space—potentially revealing clusters, trajectories of disease severity, and subtle structural patterns that a CNN cannot expose.<br>

<img width="700" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/90b5f114-aeb5-4553-a4d3-51cc943ecd94" /><br>

**To further understand the power of latent space representations, I implemented a framework for interpretable deep representation learning, disentanglement, and classification of Alzheimer’s disease from MRI. The second half of the projects implements a latent-split convolutional autoencoder that simultaneously:**
 a. Reconstructs MRI images<br>
 b.Predicts Alzheimer’s disease stage (4 classes)<br>
 c. Separates latent space into disease-related and disease-independent components<br>
 d.Visualizes latent structure using PCA and t-SNE<br>

Unlike traditional CNN classifiers, this framework not only predicts Alzheimer’s severity but also reveals how the model internally represents the brain, creating a more interpretable, compressed, and biologically meaningful latent space.<br>

**Architecture** <br>
<img width="1024" height="1536" alt="architecture" src="https://github.com/user-attachments/assets/37be1d34-63fc-4ee1-a277-474e25f048d2" />


**Interpretation of Latent Split**<br>
a.z_cls (16 dims) learns disease-specific signals such as hippocampal atrophy or ventricular enlargement<br>
b.z_corr (48 dims) learns general brain morphology but is discouraged from encoding Alzheimer severity.<br>
c.Reconstruction forces the model to use both, providing a complete structural code for each MRI slice.<br>

<img width="700" height="600" alt="tSARE_plot" src="https://github.com/user-attachments/assets/1a349826-e6a3-4c15-aeb0-1cb859f0f8a7" /><br>
<img width="700" height="600" alt="PCA_plot" src="https://github.com/user-attachments/assets/8ed1a2cf-9deb-426a-bda9-93ec18c5ec2e" /><br>
<img width="1442" height="400" alt="Loss plots" src="https://github.com/user-attachments/assets/ddf32c68-e6ae-418f-bcf7-4c39ce269db2" /><br>





