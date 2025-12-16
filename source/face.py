import torch
import cv2
from faceid.face_embedder import FaceNetRetinaEmbedder
from faceid.contrastive_siamese import Siamese

embedder = FaceNetRetinaEmbedder(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    fix_upright_rot90=False  # you already have cropped faces
)

siamese = Siamese(device='cuda' if torch.cuda.is_available() else 'cpu')  
siamese.load(r"faceid/contrastive_best/siamese_model.pt")
siamese.load_prototypes(r"faceid/contrastive_best/prototypes.csv")

def face_recognition(face_bgr):
    """
    Input:
        face_bgr: cropped face image (H, W, 3), BGR (from OpenCV)
    Output:
        512-D L2-normalized FaceNet embedding (torch.Tensor)
    """
    if face_bgr is None or face_bgr.size == 0:
        return None

    # Resize to FaceNet input size
    face_bgr = cv2.resize(face_bgr, (160, 160))

    # Use the ALREADY-developed function
    x = embedder._to_facenet_tensor(face_bgr).to(embedder.device)

    # Forward pass
    facenet_emb = embedder.model(x).squeeze(0)
    
    # Compare embedding with prototypes
    result = siamese.predict(facenet_emb)
    
    return result

    
