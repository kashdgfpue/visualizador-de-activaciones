import tkinter as tk
from tkinter import messagebox
from tkinter import ttk, filedialog
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageTk, ImageOps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import cv2
import hashlib
import json
import zlib
import base64
from skimage.metrics import structural_similarity as ssim
import pywt
from sklearn.decomposition import PCA
import warnings
from matplotlib.figure import Figure
from tkinter import ttk, filedialog, messagebox  # Added messagebox here
warnings.filterwarnings("ignore")

class UltraFidelityEssenceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("🌌 Ultra-Fidelity Essence System")
        self.root.geometry("1600x1000")

        # Configuración
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image = None
        self.essence_data = None
        self.reconstruction_artifacts = []

        # Variables de estado - MOVED BEFORE setup_ui()
        self.reconstruction_quality = tk.DoubleVar(value=0.95)
        self.visualization_mode = tk.StringVar(value="essence_3d")
        self.show_artifacts = tk.BooleanVar(value=True)

        # Modelos
        self.encoder = self.build_encoder().to(self.device)
        self.decoder = self.build_decoder().to(self.device)
        self.feature_extractor = self.build_feature_extractor().to(self.device)

        # Interfaz
        self.setup_ui()
        self.load_pretrained_weights()

        # Variables de estado
        self.reconstruction_quality = tk.DoubleVar(value=0.95)
        self.visualization_mode = tk.StringVar(value="essence_3d")
        self.show_artifacts = tk.BooleanVar(value=True)
        
    def setup_ui(self):
        """Configura la interfaz de usuario avanzada"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configurar colores
        self.bg_color = "#2e3440"
        self.fg_color = "#d8dee9"
        self.accent_color = "#5e81ac"
        
        self.style.configure('.', background=self.bg_color, foreground=self.fg_color)
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, foreground=self.fg_color)
        self.style.configure('TButton', background=self.accent_color, foreground='white')
        self.style.configure('TCombobox', fieldbackground='white')
        self.style.configure('TNotebook', background=self.bg_color)
        self.style.configure('TNotebook.Tab', background=self.bg_color, padding=[10, 5])
        
        # Paneles principales
        self.main_panel = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_panel.pack(fill=tk.BOTH, expand=True)
        
        # Panel de control (30% ancho)
        self.control_frame = ttk.Frame(self.main_panel, width=450, padding=15)
        self.main_panel.add(self.control_frame, weight=0)
        
        # Panel de visualización (70% ancho)
        self.vis_frame = ttk.Frame(self.main_panel, padding=10)
        self.main_panel.add(self.vis_frame, weight=1)
        
        # Controles
        self.setup_controls()
        
        # Visualización inicial
        self.setup_visualization()
    
    def setup_controls(self):
        """Configura los controles avanzados"""
        notebook = ttk.Notebook(self.control_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de Procesamiento
        process_tab = ttk.Frame(notebook)
        notebook.add(process_tab, text="Procesamiento")
        
        # Grupo de Carga
        load_group = ttk.LabelFrame(process_tab, text="Carga de Imagen", padding=10)
        load_group.pack(fill=tk.X, pady=5)
        
        ttk.Button(load_group, text="Cargar Imagen", command=self.load_image).pack(fill=tk.X)
        ttk.Button(load_group, text="Capturar de Cámara", command=self.capture_from_camera).pack(fill=tk.X, pady=5)
        
        # Grupo de Calidad
        quality_group = ttk.LabelFrame(process_tab, text="Control de Calidad", padding=10)
        quality_group.pack(fill=tk.X, pady=5)
        
        ttk.Scale(quality_group, from_=0.7, to=1.0, variable=self.reconstruction_quality, 
                 orient=tk.HORIZONTAL, command=lambda e: self.update_quality_display()).pack(fill=tk.X)
        self.quality_label = ttk.Label(quality_group, text="Calidad: 95%")
        self.quality_label.pack()
        
        # Grupo de Procesamiento
        action_group = ttk.LabelFrame(process_tab, text="Acciones", padding=10)
        action_group.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_group, text="Extraer Esencia", command=self.extract_essence).pack(fill=tk.X)
        ttk.Button(action_group, text="Reconstruir Imagen", command=self.reconstruct_image).pack(fill=tk.X, pady=5)
        
        # Pestaña de Visualización
        view_tab = ttk.Frame(notebook)
        notebook.add(view_tab, text="Visualización")
        
        # Modos de Visualización
        view_group = ttk.LabelFrame(view_tab, text="Modos de Visualización", padding=10)
        view_group.pack(fill=tk.BOTH, expand=True)
        
        views = [
            ("Esencia 3D", "essence_3d"),
            ("Mapa Térmico", "heatmap"),
            ("Wavelet Multinivel", "wavelet"),
            ("PCA Componentes", "pca"),
            ("Red Neuronal", "network"),
            ("Artefactos", "artifacts")
        ]
        
        for text, mode in views:
            ttk.Radiobutton(view_group, text=text, variable=self.visualization_mode, 
                          value=mode, command=self.visualize_essence).pack(anchor=tk.W)
        
        ttk.Checkbutton(view_group, text="Mostrar Artefactos", variable=self.show_artifacts,
                       command=self.visualize_essence).pack(anchor=tk.W, pady=5)
        
        # Pestaña de Exportación
        export_tab = ttk.Frame(notebook)
        notebook.add(export_tab, text="Exportación")
        
        export_group = ttk.LabelFrame(export_tab, text="Opciones de Exportación", padding=10)
        export_group.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(export_group, text="Guardar Esencia", command=self.save_essence).pack(fill=tk.X)
        ttk.Button(export_group, text="Cargar Esencia", command=self.load_essence).pack(fill=tk.X, pady=5)
        ttk.Button(export_group, text="Exportar Reporte", command=self.export_report).pack(fill=tk.X)
        ttk.Button(export_group, text="Comparar Calidad", command=self.compare_quality).pack(fill=tk.X, pady=5)
    
    def update_quality_display(self):
        """Actualiza la visualización de calidad"""
        quality = int(self.reconstruction_quality.get() * 100)
        self.quality_label.config(text=f"Calidad: {quality}%")
    
    def build_encoder(self):
        """Codificador mejorado con atención y residuos"""
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(channels)
                self.bn2 = nn.BatchNorm2d(channels)
                
            def forward(self, x):
                residual = x
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.bn2(self.conv2(x))
                x += residual
                return F.relu(x)
        
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            ResidualBlock(64),
            ResidualBlock(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            ResidualBlock(128),
            ResidualBlock(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            ResidualBlock(256),
            ResidualBlock(256),
            
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.ReLU()
        )
    
    def build_decoder(self):
        """Decodificador mejorado con upsampling aprendido"""
        class UpsampleBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                self.bn = nn.BatchNorm2d(out_channels)
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
            def forward(self, x):
                x = self.upsample(x)
                x = F.relu(self.bn(self.conv(x)))
                return x
        
        return nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            UpsampleBlock(512, 256),
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 64),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 3, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def build_feature_extractor(self):
        """Extractor de características con capas intermedias"""
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        return torch.nn.Sequential(*list(model.children())[:-2])
    
    def load_pretrained_weights(self):
        """Simula la carga de pesos preentrenados"""
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
    
    def load_image(self):
        """Carga una imagen con diálogo avanzado"""
        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen de alta resolución",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"), 
                      ("Todos los archivos", "*.*")])
        
        if filepath:
            try:
                self.current_image = Image.open(filepath).convert('RGB')
                self.show_image(self.current_image, "Imagen Original")
            except Exception as e:
                self.show_error(f"No se pudo cargar la imagen: {str(e)}")
    
    def capture_from_camera(self):
        """Captura desde cámara con configuración avanzada"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(frame)
            self.show_image(self.current_image, "Captura de Cámara")
        else:
            self.show_error("No se pudo capturar imagen de la cámara")
    
    def preprocess_image(self, image):
        """Preprocesamiento avanzado con normalización adaptativa"""
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(self.device)
    
    def extract_essence(self):
        """Extrae la esencia con múltiples niveles de representación"""
        if self.current_image is None:
            self.show_error("No hay imagen cargada")
            return
        
        try:
            # Preprocesamiento
            img_tensor = self.preprocess_image(self.current_image)
            
            # Extracción de características
            with torch.no_grad():
                # Codificación principal
                encoded = self.encoder(img_tensor)
                
                # Características semánticas profundas
                features = self.feature_extractor(img_tensor)
                
                # Hash perceptual
                img_hash = self.calculate_perceptual_hash(self.current_image)
                
                # Datos wavelet
                wavelet_coeffs = self.calculate_wavelet_transform(self.current_image)
                
                # Compresión adaptativa
                compressed_essence = self.compress_essence(encoded, features, wavelet_coeffs)
                
                # Guardar datos
                self.essence_data = {
                    'encoded': encoded,
                    'features': features,
                    'wavelet': wavelet_coeffs,
                    'hash': img_hash,
                    'quality': float(self.reconstruction_quality.get()),
                    'compressed': compressed_essence,
                    'original_size': self.current_image.size
                }
                
                # Generar artefactos de reconstrucción
                self.generate_reconstruction_artifacts(encoded)
            
            # Visualización
            self.visualize_essence()
            
        except Exception as e:
            self.show_error(f"Error al extraer esencia: {str(e)}")

    def calculate_perceptual_hash(self, image):
        """Hash perceptual avanzado"""
        # Reducción de tamaño
        img_small = image.resize((32, 32), Image.LANCZOS).convert('L')
        img_np = np.array(img_small, dtype=np.float32)

        # Flatten and normalize
        img_flat = img_np.flatten()
        img_flat = (img_flat - img_flat.mean()) / (img_flat.std() + 1e-8)

        # Use DCT instead of PCA for perceptual hashing
        dct = cv2.dct(img_np.astype(np.float32))
        top_left = dct[:8, :8].flatten()

        # Calculate hash from DCT coefficients
        median = np.median(top_left)
        hash_str = ''.join(['1' if x > median else '0' for x in top_left])
        return hashlib.sha256(hash_str.encode()).hexdigest()[:24]

    def calculate_wavelet_transform(self, image):
        """Transformada wavelet multinivel con manejo de tipos"""
        # Convert to grayscale numpy array with proper type
        img_np = np.array(image.convert('L'), dtype=np.float32)
        
        # Perform wavelet transform
        coeffs = pywt.wavedec2(img_np, 'db4', level=3)
        
        # Ensure all coefficients are properly typed
        processed_coeffs = []
        for c in coeffs:
            if isinstance(c, tuple):
                # For detail coefficients (tuple of arrays)
                processed_coeffs.append(tuple(
                    arr.astype(np.float32) for arr in c
                ))
            else:
                # For approximation coefficient (single array)
                processed_coeffs.append(c.astype(np.float32))
        
        return processed_coeffs
    

    def compress_essence(self, encoded, features, wavelet_coeffs):
        """Compresión adaptativa con múltiples componentes"""
        # Preparar datos
        encoded_np = encoded.squeeze(0).cpu().numpy()
        features_np = features.squeeze(0).cpu().numpy()
        
        # Comprimir wavelet
        wavelet_bytes = pywt.array_to_coeffs(
            np.zeros_like(wavelet_coeffs[0]), 
            wavelet_coeffs[1:], 
            output_format='wavedec2'
        )
        wavelet_compressed = zlib.compress(wavelet_bytes)
        
        # Comprimir tensores
        quality = self.reconstruction_quality.get()
        compress_level = int(9 * quality)
        
        combined = np.concatenate([
            encoded_np.flatten(),
            features_np.flatten()
        ])
        tensor_compressed = zlib.compress(combined.tobytes(), level=compress_level)
        
        # Estructura comprimida
        compressed_data = {
            'tensors': base64.b64encode(tensor_compressed).decode('utf-8'),
            'wavelet': base64.b64encode(wavelet_compressed).decode('utf-8'),
            'metadata': {
                'encoded_shape': encoded_np.shape,
                'features_shape': features_np.shape,
                'wavelet_type': 'db4',
                'compression_level': compress_level
            }
        }
        
        return compressed_data
    
    def generate_reconstruction_artifacts(self, encoded):
        """Genera artefactos para guiar la reconstrucción"""
        self.reconstruction_artifacts = []
        encoded_np = encoded.squeeze(0).cpu().numpy()
        
        # 1. Hash de canales principales
        for i in range(0, encoded_np.shape[0], 64):
            channel_block = encoded_np[i:i+64]
            block_hash = hashlib.sha256(channel_block.tobytes()).hexdigest()[:12]
            self.reconstruction_artifacts.append(('channel_hash', block_hash))
        
        # 2. Estadísticas de características
        mean = encoded_np.mean()
        std = encoded_np.std()
        self.reconstruction_artifacts.append(('stats', f'μ={mean:.4f}, σ={std:.4f}'))
        
        # 3. Componentes principales
        pca = PCA(n_components=3)
        flattened = encoded_np.reshape(encoded_np.shape[0], -1).T
        pca.fit(flattened)
        self.reconstruction_artifacts.append(('pca', pca.explained_variance_ratio_))
    
    def reconstruct_image(self):
        """Reconstrucción de alta fidelidad con verificación"""
        if self.essence_data is None:
            self.show_error("No hay esencia disponible")
            return
        
        try:
            # Descompresión
            tensor_data = base64.b64decode(self.essence_data['compressed']['tensors'])
            wavelet_data = base64.b64decode(self.essence_data['compressed']['wavelet'])
            
            # Descomprimir tensores
            tensor_bytes = zlib.decompress(tensor_data)
            metadata = self.essence_data['compressed']['metadata']
            
            # Reconstruir arrays
            total_size = (np.prod(metadata['encoded_shape']) + 
                         np.prod(metadata['features_shape']))
            decompressed = np.frombuffer(tensor_bytes, dtype=np.float32, count=total_size)
            
            # Separar componentes
            encoded_size = np.prod(metadata['encoded_shape'])
            encoded_flat = decompressed[:encoded_size]
            features_flat = decompressed[encoded_size:]
            
            encoded_np = encoded_flat.reshape(metadata['encoded_shape'])
            features_np = features_flat.reshape(metadata['features_shape'])
            
            # Descomprimir wavelet
            wavelet_bytes = zlib.decompress(wavelet_data)
            wavelet_coeffs = pywt.coeffs_to_array(pywt.array_to_coeffs(
                np.zeros_like(self.essence_data['wavelet'][0]), 
                self.essence_data['wavelet'][1:], 
                output_format='wavedec2'
            ))
            
            # Convertir a tensores
            encoded = torch.from_numpy(encoded_np).unsqueeze(0).to(self.device)
            features = torch.from_numpy(features_np).unsqueeze(0).to(self.device)
            
            # Verificar artefactos
            self.verify_reconstruction_artifacts(encoded)
            
            # Reconstrucción con refinamiento iterativo
            reconstructed = self.iterative_reconstruction(encoded)
            
            # Post-procesamiento
            reconstructed = self.postprocess_reconstruction(reconstructed)
            
            # Guardar resultado
            self.essence_data['reconstructed'] = reconstructed
            self.show_image(reconstructed, "Imagen Reconstruida")
            
        except Exception as e:
            self.show_error(f"Error en reconstrucción: {str(e)}")
    
    def iterative_reconstruction(self, encoded):
        """Reconstrucción iterativa para mejor calidad"""
        current = self.decoder(encoded)
        
        # Refinamiento en 3 pasos
        for i in range(3):
            residual = self.encoder(current) - encoded
            correction = self.decoder(residual)
            current = torch.clamp(current + correction * 0.3, 0, 1)
        
        return current.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    def postprocess_reconstruction(self, reconstructed):
        """Post-procesamiento para mejorar calidad visual"""
        # Convertir a PIL Image
        reconstructed = (reconstructed * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(reconstructed)
        
        # Aplicar suavizado adaptativo
        if self.reconstruction_quality.get() > 0.9:
            img = img.filter(ImageFilter.SMOOTH_MORE)
        
        # Ajustar tamaño original
        if 'original_size' in self.essence_data:
            img = img.resize(self.essence_data['original_size'], Image.LANCZOS)
        
        return img
    
    def verify_reconstruction_artifacts(self, encoded):
        """Verifica los artefactos de reconstrucción"""
        encoded_np = encoded.squeeze(0).cpu().numpy()
        issues = 0
        
        # Verificar hashes de canales
        for i, (art_type, art_value) in enumerate(self.reconstruction_artifacts):
            if art_type == 'channel_hash':
                channel_block = encoded_np[i*64:(i+1)*64]
                current_hash = hashlib.sha256(channel_block.tobytes()).hexdigest()[:12]
                if current_hash != art_value:
                    issues += 1
        
        # Verificar estadísticas
        if issues == 0:
            current_mean = encoded_np.mean()
            current_std = encoded_np.std()
            _, stats_str = self.reconstruction_artifacts[-2]
            orig_mean = float(stats_str.split('μ=')[1].split(',')[0])
            orig_std = float(stats_str.split('σ=')[1])
            
            if (abs(current_mean - orig_mean) > 0.1 * orig_mean or 
                abs(current_std - orig_std) > 0.1 * orig_std):
                issues += 1
        
        if issues > 0:
            self.show_warning(f"Se detectaron {issues} inconsistencias en la reconstrucción. La calidad puede estar afectada.")
    
    def visualize_essence(self):
        """Visualización avanzada de la esencia"""
        if self.essence_data is None:
            return
            
        self.fig.clf()
        
        mode = self.visualization_mode.get()
        
        if mode == "essence_3d":
            self.show_3d_essence()
        elif mode == "heatmap":
            self.show_heatmap()
        elif mode == "wavelet":
            self.show_wavelet_decomposition()
        elif mode == "pca":
            self.show_pca_components()
        elif mode == "network":
            self.show_network_architecture()
        elif mode == "artifacts":
            self.show_reconstruction_artifacts()
        
        self.canvas.draw()
    
    def show_3d_essence(self):
        """Visualización 3D de la esencia codificada"""
        encoded = self.essence_data['encoded'].squeeze(0).cpu().numpy()
        
        # Tomar los primeros 3 canales para RGB
        if encoded.shape[0] < 3:
            encoded = np.pad(encoded, ((0, 3 - encoded.shape[0]), (0, 0), (0, 0)))
        
        # Normalizar
        encoded = (encoded - encoded.min()) / (encoded.max() - encoded.min())
        
        # Crear figura 3D
        ax = self.fig.add_subplot(111, projection='3d')
        
        # Solo mostrar una muestra de los datos para mejor rendimiento
        step = 2
        x, y = np.mgrid[0:encoded.shape[1]:step, 0:encoded.shape[2]:step]
        
        # Graficar cada canal
        for i in range(min(3, encoded.shape[0])):
            z = encoded[i, ::step, ::step]
            ax.plot_surface(x, y, z*(i+1), color=plt.cm.viridis(i/3), alpha=0.7)
        
        ax.set_title("Representación 3D de la Esencia Codificada")
        ax.set_xlabel("Ancho")
        ax.set_ylabel("Alto")
        ax.set_zlabel("Intensidad")
    
    def show_heatmap(self):
        """Mapa térmico de características"""
        features = self.essence_data['features'].squeeze(0).cpu().numpy()
        avg_features = np.mean(features, axis=0)
        
        ax = self.fig.add_subplot(111)
        im = ax.imshow(avg_features, cmap='inferno')
        
        ax.set_title("Mapa Térmico de Características Promedio")
        ax.axis('off')
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def show_wavelet_decomposition(self):
        """Visualización wavelet multinivel"""
        coeffs = self.essence_data['wavelet']
        titles = ['Aproximación', 'Detalle 1', 'Detalle 2', 'Detalle 3']
        
        for i, (coeff, title) in enumerate(zip(coeffs, titles)):
            ax = self.fig.add_subplot(2, 2, i+1)
            
            if isinstance(coeff, tuple):
                # Para coeficientes de detalle
                c = np.sqrt(coeff[0]**2 + coeff[1]**2 + coeff[2]**2)
                ax.imshow(c, cmap='viridis')
            else:
                # Para coeficiente de aproximación
                ax.imshow(coeff, cmap='gray')
            
            ax.set_title(title)
            ax.axis('off')
        
        self.fig.suptitle("Descomposición Wavelet Multinivel", y=1.02)
    
    def show_pca_components(self):
        """Componentes principales de la esencia"""
        encoded = self.essence_data['encoded'].squeeze(0).cpu().numpy()
        flattened = encoded.reshape(encoded.shape[0], -1).T
        
        pca = PCA(n_components=3)
        components = pca.fit_transform(flattened)
        components = components.T.reshape(3, *encoded.shape[1:])
        
        titles = ['Componente 1', 'Componente 2', 'Componente 3']
        
        for i in range(3):
            ax = self.fig.add_subplot(1, 3, i+1)
            ax.imshow(components[i], cmap='coolwarm')
            ax.set_title(titles[i])
            ax.axis('off')
        
        self.fig.suptitle(f"Componentes Principales (Varianza explicada: {np.sum(pca.explained_variance_ratio_):.2%})", y=1.05)
    
    def show_network_architecture(self):
        """Visualización de la arquitectura de red"""
        ax = self.fig.add_subplot(111)
        
        # Información de la arquitectura
        encoder_info = "Codificador:\n" + "\n".join(
            f"{i}. {layer.__class__.__name__}" 
            for i, layer in enumerate(self.encoder.children())
        )
        
        decoder_info = "\n\nDecodificador:\n" + "\n".join(
            f"{i}. {layer.__class__.__name__}" 
            for i, layer in enumerate(self.decoder.children())
        )
        
        ax.text(0.1, 0.5, encoder_info + decoder_info, 
               fontfamily='monospace', fontsize=10, va='center')
        ax.set_title("Arquitectura de la Red Neuronal")
        ax.axis('off')
    
    def show_reconstruction_artifacts(self):
        """Muestra los artefactos de reconstrucción"""
        ax = self.fig.add_subplot(111)
        
        info = ["Artefactos de Reconstrucción:", ""]
        
        for art_type, art_value in self.reconstruction_artifacts:
            if art_type == 'channel_hash':
                info.append(f"Hash Canal: {art_value}")
            elif art_type == 'stats':
                info.append(f"Estadísticas: {art_value}")
            elif art_type == 'pca':
                info.append(f"PCA: {np.array2string(art_value, precision=2)}")
        
        ax.text(0.1, 0.5, "\n".join(info), fontfamily='monospace', fontsize=12)
        ax.set_title("Huellas para Reconstrucción", pad=20)
        ax.axis('off')
    
    def show_image(self, image, title="Imagen"):
        """Muestra una imagen con barra de herramientas"""
        for widget in self.vis_frame.winfo_children():
            widget.destroy()
        
        self.fig = Figure(figsize=(10, 8), dpi=100)
        ax = self.fig.add_subplot(111)
        ax.imshow(image)
        ax.set_title(title, pad=20)
        ax.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.vis_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, self.vis_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def save_essence(self):
        """Guarda la esencia con todos los metadatos"""
        if self.essence_data is None:
            self.show_error("No hay esencia para guardar")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".ufes",
            filetypes=[("Archivo Ultra-Fidelity Essence", "*.ufes"), ("Todos los archivos", "*.*")])
        
        if filepath:
            try:
                # Preparar datos para guardar
                save_data = {
                    'essence': self.essence_data['compressed'],
                    'metadata': {
                        'original_size': self.essence_data['original_size'],
                        'quality': self.essence_data['quality'],
                        'hash': self.essence_data['hash'],
                        'model': 'UltraFidelity_v3',
                        'device': str(self.device),
                        'timestamp': str(datetime.now())
                    },
                    'artifacts': self.reconstruction_artifacts
                }
                
                with open(filepath, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                self.show_info(f"Esencia guardada en:\n{filepath}")
                
            except Exception as e:
                self.show_error(f"Error al guardar: {str(e)}")
    
    def load_essence(self):
        """Carga una esencia guardada"""
        filepath = filedialog.askopenfilename(
            filetypes=[("Archivo Ultra-Fidelity Essence", "*.ufes"), ("Todos los archivos", "*.*")])
        
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    load_data = json.load(f)
                
                # Reconstruir datos de esencia
                self.essence_data = {
                    'compressed': load_data['essence'],
                    'original_size': load_data['metadata']['original_size'],
                    'quality': load_data['metadata']['quality'],
                    'hash': load_data['metadata']['hash'],
                    'reconstructed': None
                }
                
                self.reconstruction_artifacts = load_data.get('artifacts', [])
                self.reconstruction_quality.set(load_data['metadata']['quality'])
                self.update_quality_display()
                
                self.show_info("Esencia cargada exitosamente")
                self.visualize_essence()
                
            except Exception as e:
                self.show_error(f"Error al cargar: {str(e)}")
    
    def export_report(self):
        """Exporta un reporte completo en PDF"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("Archivo PDF", "*.pdf"), ("Todos los archivos", "*.*")])
        
        if filepath:
            try:
                from matplotlib.backends.backend_pdf import PdfPages
                
                with PdfPages(filepath) as pdf:
                    # Página 1: Imagen original y reconstruida
                    plt.figure(figsize=(10, 8))
                    plt.suptitle("Reporte de Esencia - Comparación", y=0.95)
                    
                    if self.current_image:
                        plt.subplot(121)
                        plt.imshow(self.current_image)
                        plt.title("Original")
                        plt.axis('off')
                    
                    if 'reconstructed' in self.essence_data and self.essence_data['reconstructed']:
                        plt.subplot(122)
                        plt.imshow(self.essence_data['reconstructed'])
                        plt.title("Reconstruida")
                        plt.axis('off')
                    
                    pdf.savefig()
                    plt.close()
                    
                    # Página 2: Visualizaciones de esencia
                    plt.figure(figsize=(10, 8))
                    plt.suptitle("Reporte de Esencia - Visualizaciones", y=0.95)
                    
                    # Mostrar diferentes visualizaciones
                    visualizations = ['essence_3d', 'heatmap', 'wavelet', 'pca']
                    for i, vis in enumerate(visualizations, 1):
                        self.visualization_mode.set(vis)
                        self.visualize_essence()
                        pdf.savefig(self.fig)
                        plt.close()
                
                self.show_info(f"Reporte exportado a:\n{filepath}")
                
            except Exception as e:
                self.show_error(f"Error al exportar reporte: {str(e)}")
    
    def compare_quality(self):
        """Comparación detallada de calidad"""
        if self.current_image is None or 'reconstructed' not in self.essence_data:
            self.show_error("Necesitas una imagen original y reconstruida")
            return
        
        try:
            # Convertir imágenes a arrays
            original = np.array(self.current_image.resize((512, 512)).convert('RGB'))
            reconstructed = np.array(self.essence_data['reconstructed'].resize((512, 512)).convert('RGB'))
            
            # Calcular métricas
            mse = np.mean((original - reconstructed) ** 2)
            psnr = 10 * np.log10(255**2 / mse)
            ssim_val = ssim(original, reconstructed, multichannel=True, 
                           data_range=original.max() - original.min())
            
            # Crear figura de comparación
            self.fig.clf()
            
            # Imágenes
            ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
            ax1.imshow(self.current_image)
            ax1.set_title("Original")
            ax1.axis('off')
            
            ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
            ax2.imshow(Image.fromarray(reconstructed))
            ax2.set_title("Reconstruida")
            ax2.axis('off')
            
            # Diferencia
            ax3 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
            difference = np.abs(original - reconstructed)
            ax3.imshow(difference, cmap='hot')
            ax3.set_title("Diferencia")
            ax3.axis('off')
            
            # Métricas
            metrics_text = f"MSE: {mse:.2f}\nPSNR: {psnr:.2f} dB\nSSIM: {ssim_val:.4f}"
            self.fig.text(0.7, 0.3, metrics_text, fontsize=12, 
                         bbox=dict(facecolor='white', alpha=0.8))
            
            self.fig.suptitle("Análisis de Calidad de Reconstrucción", fontsize=14)
            self.canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error en comparación: {str(e)}")
    
    def setup_visualization(self):
        """Configura el área de visualización inicial"""
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.subplots_adjust(top=0.9)
        self.fig.suptitle("Ultra-Fidelity Essence System", fontsize=16)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.vis_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, self.vis_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mensaje inicial
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Cargue una imagen para comenzar", 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    def show_error(self, message):
        """Muestra un mensaje de error"""
        messagebox.showerror("Error", message)
    
    def show_warning(self, message):
        """Muestra un mensaje de advertencia"""
        messagebox.showwarning("Advertencia", message)
    
    def show_info(self, message):
        """Muestra un mensaje informativo"""
        messagebox.showinfo("Información", message)

# Punto de entrada de la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = UltraFidelityEssenceSystem(root)
    root.mainloop()