import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2
import io
import base64
import json
import os

class AdvancedEssenceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("🧿 Advanced Essence System")
        self.root.geometry("1400x900")
        
        # Configuración
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image = None
        self.essence_data = None
        
        # Modelos
        self.encoder = self.build_encoder().to(self.device)
        self.decoder = self.build_decoder().to(self.device)
        self.feature_extractor = self.build_feature_extractor().to(self.device)
        
        # Interfaz
        self.setup_ui()
        self.load_models()
        
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Paneles principales
        self.main_panel = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_panel.pack(fill=tk.BOTH, expand=True)
        
        # Panel de control
        self.control_frame = ttk.Frame(self.main_panel, width=300, padding=10)
        self.main_panel.add(self.control_frame, weight=0)
        
        # Panel de visualización
        self.vis_frame = ttk.Frame(self.main_panel, padding=10)
        self.main_panel.add(self.vis_frame, weight=1)
        
        # Controles
        self.setup_controls()
        
    def setup_controls(self):
        """Configura los controles de la interfaz"""
        # Grupo de carga
        load_group = ttk.LabelFrame(self.control_frame, text="Carga de Imagen", padding=10)
        load_group.pack(fill=tk.X, pady=5)
        
        ttk.Button(load_group, text="Cargar Imagen", command=self.load_image).pack(fill=tk.X)
        ttk.Button(load_group, text="Capturar de Cámara", command=self.capture_from_camera).pack(fill=tk.X, pady=5)
        
        # Grupo de procesamiento
        process_group = ttk.LabelFrame(self.control_frame, text="Procesamiento", padding=10)
        process_group.pack(fill=tk.X, pady=5)
        
        ttk.Button(process_group, text="Extraer Esencia", command=self.extract_essence).pack(fill=tk.X)
        ttk.Button(process_group, text="Reconstruir Imagen", command=self.reconstruct_image).pack(fill=tk.X, pady=5)
        
        # Grupo de visualización
        view_group = ttk.LabelFrame(self.control_frame, text="Visualización", padding=10)
        view_group.pack(fill=tk.X, pady=5)
        
        self.view_mode = tk.StringVar(value="feature_maps")
        ttk.Radiobutton(view_group, text="Mapas de Características", 
                       variable=self.view_mode, value="feature_maps").pack(anchor=tk.W)
        ttk.Radiobutton(view_group, text="Activaciones CNN", 
                       variable=self.view_mode, value="activations").pack(anchor=tk.W)
        ttk.Radiobutton(view_group, text="Grad-CAM", 
                       variable=self.view_mode, value="gradcam").pack(anchor=tk.W)
        ttk.Radiobutton(view_group, text="Descomposición Wavelet", 
                       variable=self.view_mode, value="wavelet").pack(anchor=tk.W)
        
        # Grupo de opciones
        options_group = ttk.LabelFrame(self.control_frame, text="Opciones", padding=10)
        options_group.pack(fill=tk.X, pady=5)
        
        ttk.Button(options_group, text="Guardar Esencia", command=self.save_essence).pack(fill=tk.X)
        ttk.Button(options_group, text="Cargar Esencia", command=self.load_essence).pack(fill=tk.X, pady=5)
        ttk.Button(options_group, text="Configuración", command=self.show_settings).pack(fill=tk.X)
        
        # Visualización inicial
        self.setup_visualization()
    
    def setup_visualization(self):
        """Configura el área de visualización"""
        for widget in self.vis_frame.winfo_children():
            widget.destroy()
        
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mostrar imagen inicial
        self.show_placeholder()
    
    def show_placeholder(self):
        """Muestra un marcador de posición inicial"""
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Cargue una imagen para comenzar", 
               ha='center', va='center', fontsize=12)
        ax.axis('off')
        self.canvas.draw()
    
    def build_encoder(self):
        """Construye el modelo codificador"""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
    
    def build_decoder(self):
        """Construye el modelo decodificador"""
        return nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def build_feature_extractor(self):
        """Construye el extractor de características"""
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-2])  # Eliminar capas finales
        return model
    
    def load_models(self):
        """Carga los modelos (en una implementación real cargaría pesos guardados)"""
        pass
    
    def load_image(self):
        """Carga una imagen desde archivo"""
        filepath = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*")])
        
        if filepath:
            try:
                self.current_image = Image.open(filepath).convert('RGB')
                self.show_image(self.current_image)
            except Exception as e:
                self.show_error(f"No se pudo cargar la imagen: {str(e)}")
    
    def capture_from_camera(self):
        """Captura una imagen desde la cámara web"""
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(frame)
            self.show_image(self.current_image)
        else:
            self.show_error("No se pudo acceder a la cámara")
    
    def show_image(self, image):
        """Muestra una imagen en el área de visualización"""
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title("Imagen Original", pad=20)
        self.canvas.draw()
    
    def extract_essence(self):
        """Extrae la esencia de la imagen actual"""
        if self.current_image is None:
            self.show_error("No hay imagen cargada")
            return
        
        try:
            # Preprocesamiento
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            img_tensor = transform(self.current_image).unsqueeze(0).to(self.device)
            
            # Codificación
            with torch.no_grad():
                self.essence_data = self.encoder(img_tensor)
                features = self.feature_extractor(img_tensor)
            
            # Visualización
            self.visualize_essence(self.essence_data, features)
            
        except Exception as e:
            self.show_error(f"Error al extraer esencia: {str(e)}")
    
    def visualize_essence(self, essence, features):
        """Visualiza la esencia según el modo seleccionado"""
        self.fig.clf()
        
        if self.view_mode.get() == "feature_maps":
            self.show_feature_maps(essence)
        elif self.view_mode.get() == "activations":
            self.show_activations(features)
        elif self.view_mode.get() == "gradcam":
            self.show_gradcam(features)
        elif self.view_mode.get() == "wavelet":
            self.show_wavelet_decomposition(essence)
        
        self.canvas.draw()
    
    def show_feature_maps(self, essence):
        """Muestra los mapas de características"""
        essence = essence.squeeze(0).cpu().numpy()
        num_features = min(essence.shape[0], 16)  # Mostrar máximo 16 características
        
        rows = int(np.ceil(np.sqrt(num_features)))
        cols = int(np.ceil(num_features / rows))
        
        for i in range(num_features):
            ax = self.fig.add_subplot(rows, cols, i+1)
            ax.imshow(essence[i], cmap='viridis')
            ax.set_title(f'Canal {i+1}', fontsize=8)
            ax.axis('off')
        
        self.fig.suptitle("Mapas de Características de la Esencia", y=1.02)
        self.fig.tight_layout()
    
    def show_activations(self, features):
        """Muestra las activaciones CNN"""
        features = features.squeeze(0).cpu().numpy()
        num_features = min(features.shape[0], 16)  # Mostrar máximo 16 activaciones
        
        rows = int(np.ceil(np.sqrt(num_features)))
        cols = int(np.ceil(num_features / rows))
        
        for i in range(num_features):
            ax = self.fig.add_subplot(rows, cols, i+1)
            ax.imshow(features[i], cmap='plasma')
            ax.set_title(f'Activación {i+1}', fontsize=8)
            ax.axis('off')
        
        self.fig.suptitle("Activaciones CNN", y=1.02)
        self.fig.tight_layout()
    
    def show_gradcam(self, features):
        """Muestra un mapa de activación Grad-CAM"""
        # Implementación simplificada de Grad-CAM
        features = features.squeeze(0).cpu().numpy()
        gradcam = np.mean(features, axis=0)
        gradcam = np.maximum(gradcam, 0)
        gradcam = gradcam / gradcam.max()
        
        ax = self.fig.add_subplot(111)
        ax.imshow(gradcam, cmap='jet', alpha=0.5)
        
        # Superponer en la imagen original
        img = self.current_image.resize((gradcam.shape[1], gradcam.shape[0]))
        ax.imshow(img, alpha=0.5)
        
        ax.set_title("Grad-CAM: Regiones Importantes", pad=20)
        ax.axis('off')
    
    def show_wavelet_decomposition(self, essence):
        """Muestra una descomposición wavelet simplificada"""
        # Implementación simplificada usando diferencias
        essence = essence.squeeze(0).cpu().numpy()
        low_freq = essence[0]
        high_freq = np.mean(essence[1:], axis=0)
        
        ax1 = self.fig.add_subplot(121)
        ax1.imshow(low_freq, cmap='gray')
        ax1.set_title("Bajas Frecuencias", pad=10)
        ax1.axis('off')
        
        ax2 = self.fig.add_subplot(122)
        ax2.imshow(high_freq, cmap='gray')
        ax2.set_title("Altas Frecuencias", pad=10)
        ax2.axis('off')
        
        self.fig.suptitle("Descomposición Wavelet Simplificada", y=1.02)
    
    def reconstruct_image(self):
        """Reconstruye la imagen desde la esencia"""
        if self.essence_data is None:
            self.show_error("No hay esencia disponible para reconstrucción")
            return
        
        try:
            with torch.no_grad():
                reconstructed = self.decoder(self.essence_data)
                reconstructed = reconstructed.squeeze(0).cpu().permute(1, 2, 0).numpy()
                reconstructed = (reconstructed * 255).astype(np.uint8)
                reconstructed_img = Image.fromarray(reconstructed)
            
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.imshow(reconstructed_img)
            ax.set_title("Imagen Reconstruida", pad=20)
            ax.axis('off')
            self.canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error en reconstrucción: {str(e)}")
    
    def save_essence(self):
        """Guarda los datos de esencia"""
        if self.essence_data is None:
            self.show_error("No hay esencia para guardar")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".essence",
            filetypes=[("Archivo de Esencia", "*.essence"), ("Todos los archivos", "*.*")])
        
        if filepath:
            try:
                # Convertir a formato serializable
                essence_np = self.essence_data.squeeze(0).cpu().numpy()
                essence_bytes = essence_np.tobytes()
                essence_b64 = base64.b64encode(essence_bytes).decode('utf-8')
                
                # Crear diccionario de metadatos
                metadata = {
                    'shape': essence_np.shape,
                    'dtype': str(essence_np.dtype),
                    'data': essence_b64
                }
                
                with open(filepath, 'w') as f:
                    json.dump(metadata, f)
                
                self.show_info(f"Esencia guardada en:\n{filepath}")
                
            except Exception as e:
                self.show_error(f"Error al guardar: {str(e)}")
    
    def load_essence(self):
        """Carga datos de esencia desde archivo"""
        filepath = filedialog.askopenfilename(
            filetypes=[("Archivo de Esencia", "*.essence"), ("Todos los archivos", "*.*")])
        
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    metadata = json.load(f)
                
                # Reconstruir los datos
                essence_bytes = base64.b64decode(metadata['data'])
                essence_np = np.frombuffer(essence_bytes, dtype=metadata['dtype'])
                essence_np = essence_np.reshape(metadata['shape'])
                
                self.essence_data = torch.from_numpy(essence_np).unsqueeze(0).to(self.device)
                self.show_info("Esencia cargada exitosamente")
                
                # Visualizar la esencia cargada
                self.visualize_essence(self.essence_data, None)
                
            except Exception as e:
                self.show_error(f"Error al cargar: {str(e)}")
    
    def show_settings(self):
        """Muestra el diálogo de configuración"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Configuración")
        settings_window.geometry("400x300")
        
        ttk.Label(settings_window, text="Opciones de Visualización", font=('Helvetica', 12)).pack(pady=10)
        
        # Selector de colormap
        ttk.Label(settings_window, text="Mapa de colores:").pack()
        cmap_var = tk.StringVar(value='viridis')
        cmap_menu = ttk.Combobox(settings_window, textvariable=cmap_var, 
                                values=['viridis', 'plasma', 'magma', 'jet', 'coolwarm'])
        cmap_menu.pack(fill=tk.X, padx=50, pady=5)
        
        # Botones
        btn_frame = ttk.Frame(settings_window)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Aplicar", 
                  command=lambda: self.apply_settings(cmap_var.get())).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Cancelar", 
                  command=settings_window.destroy).pack(side=tk.LEFT, padx=10)
    
    def apply_settings(self, cmap):
        """Aplica la configuración seleccionada"""
        # En una implementación real, esto afectaría las visualizaciones
        self.show_info(f"Configuración aplicada: Mapa de colores = {cmap}")
    
    def show_error(self, message):
        """Muestra un mensaje de error"""
        tk.messagebox.showerror("Error", message)
    
    def show_info(self, message):
        """Muestra un mensaje informativo"""
        tk.messagebox.showinfo("Información", message)

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedEssenceSystem(root)
    root.mainloop()