import torch
import os
from torchvision import datasets, transforms # Asegúrate de importar transforms aquí
from PIL import Image # Asegúrate de importar Image aquí

# Importa las clases del modelo y utilidades
from src.model import VisionTransformer
from src.utils import preprocess_image_for_inference, preprocess_raw_tensor_for_inference, IMAGE_SIZE, get_mnist_transform # Importa get_mnist_transform

# predict.py
# --- Parámetros del modelo (Deben ser los mismos que en train.py) ---
PATCH_SIZE = 7
IN_CHANNELS = 1
EMBED_DIM = 128
NUM_HEADS = 8
NUM_TRANSFORMER_LAYERS = 4
NUM_CLASSES = 10

# Ruta donde se guardó el modelo
MODEL_PATH = 'models/vit_mnist_model.pth'

# Configuración del dispositivo (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cargando modelo en {device}...")

def load_trained_model():
    """Carga un modelo Vision Transformer con pesos entrenados."""
    model = VisionTransformer(IMAGE_SIZE, PATCH_SIZE, IN_CHANNELS, EMBED_DIM, 
                              NUM_HEADS, NUM_TRANSFORMER_LAYERS, NUM_CLASSES)
    model.to(device)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"El archivo del modelo no se encontró en: {MODEL_PATH}. "
                                "Por favor, entrena el modelo primero ejecutando train.py.")
                                
    # Carga los pesos entrenados
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Pone el modelo en modo de evaluación
    print("Modelo cargado exitosamente.")
    return model

def predict(model, input_tensor):
    """
    Realiza una predicción en un tensor de entrada preprocesado.
    input_tensor: Tensor de PyTorch de la imagen con forma (1, C, H, W).
    """
    with torch.no_grad():
        output_logits = model(input_tensor)
    
    probabilities = torch.softmax(output_logits, dim=1)
    predicted_prob, predicted_class_tensor = torch.max(probabilities, dim=1)
    
    return predicted_class_tensor.item(), predicted_prob.item()


if __name__ == '__main__':
    trained_model = load_trained_model()

    # --- Ejemplo de inferencia con una imagen del dataset de prueba ---
    print("\n--- Inferencia con una imagen del dataset de prueba (primera imagen) ---")
    
    # IMPORTANTE: Aquí vamos a cargar el dataset DE NUEVO pero aplicando las transformaciones de entrenamiento
    # Esto asegura que la imagen se convierta a tensor y se normalice de la misma manera que en el entrenamiento
    test_dataset_transformed = datasets.MNIST(root='./data', train=False, download=True, transform=get_mnist_transform())
    
    # Ahora sample_image_transformed ya es un tensor de PyTorch preprocesado
    sample_image_transformed, true_label_transformed = test_dataset_transformed[0]
    print(f"Etiqueta real de la imagen de ejemplo: {true_label_transformed}")

    # Pasa el tensor directamente a preprocess_raw_tensor_for_inference
    # Esta función ahora solo asegura la dimensión del lote y el dispositivo.
    input_tensor_for_prediction = preprocess_raw_tensor_for_inference(sample_image_transformed, device)
    
    predicted_class, predicted_prob = predict(trained_model, input_tensor_for_prediction)
    print(f"Clase predicha: {predicted_class}")
    print(f"Probabilidad de la clase predicha: {predicted_prob:.4f}")

    # --- Ejemplo de inferencia con una imagen de archivo (PNG/JPG) ---
    # Generar y guardar una imagen de prueba si no existe (requiere matplotlib)
    try:
        import matplotlib.pyplot as plt
        # Asegúrate de que tienes `sample_digit_0.png` en la misma carpeta o especifica la ruta
        if not os.path.exists('sample_digit_0.png'):
            # Genera y guarda una imagen de prueba (ej. el primer dígito del MNIST)
            # test_dataset.data[0] es un tensor de torch.uint8 de (H, W)
            original_img_numpy = test_dataset.data[0].numpy() # Convertir a numpy para imshow
            plt.imshow(original_img_numpy, cmap='gray')
            plt.axis('off')
            plt.savefig('sample_digit_0.png', bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()
            print("Imagen de prueba 'sample_digit_0.png' generada.")
    except ImportError:
        print("matplotlib no está instalado. No se puede generar la imagen de prueba.")
    
    image_file_path = 'prueba5.png'
    if os.path.exists(image_file_path):
        input_tensor_from_file = preprocess_image_for_inference(image_file_path, device)
        predicted_class_file, predicted_prob_file = predict(trained_model, input_tensor_from_file)
        print(f"Predicción para '{image_file_path}': Clase {predicted_class_file}, Probabilidad {predicted_prob_file:.4f}")
    else:
        print(f"No se encontró la imagen de prueba '{image_file_path}'. Omite la inferencia de archivo.")
