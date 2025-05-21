import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import librosa
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import threading
import queue

# Configuración
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================
# Modelo Avanzado de Extracción de Esencia (Versión Corregida)
# ==============================================
class EsenciaExtractor:
    def __init__(self):
        self.device = device
        self._cargar_modelos()
        self._preparar_transformaciones()
        
    def _cargar_modelos(self):
        # CNN 3D con dimensiones ajustadas
        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        ).to(self.device).eval()
        
        # LSTM para características temporales
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, 
                           num_layers=2, batch_first=True).to(self.device).eval()
        
        # Transformer para texto
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.transformer = AutoModel.from_pretrained('distilbert-base-uncased').to(self.device).eval()
        
    def _preparar_transformaciones(self):
        self.img_transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Aumentamos el tamaño de entrada
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def extraer_esencia(self, input_data, tipo):
        if tipo == 'imagen':
            return self._procesar_imagen(input_data)
        elif tipo == 'video':
            return self._procesar_video(input_data)
        elif tipo == 'texto':
            return self._procesar_texto(input_data)
        elif tipo == 'audio':
            return self._procesar_audio(input_data)
        else:
            raise ValueError(f"Tipo no soportado: {tipo}")
    
    def _procesar_imagen(self, img):
        try:
            if isinstance(img, str):  # Ruta de archivo
                img = Image.open(img).convert('RGB')
            elif isinstance(img, np.ndarray):  # Frame de cámara
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            img_tensor = self.img_transform(img).unsqueeze(0).to(self.device)
            img_volumen = img_tensor.repeat(8, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)
            
            with torch.no_grad():
                features_3d = self.cnn3d(img_volumen).cpu().numpy()
            
            return self._binarizar_features(features_3d)
        except Exception as e:
            raise ValueError(f"Error procesando imagen: {str(e)}")
    
    def _procesar_video(self, video_source):
        try:
            frames = []
            if isinstance(video_source, str):  # Ruta de video
                cap = cv2.VideoCapture(video_source)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Muestreamos 8 frames equidistantes
                frame_indices = np.linspace(0, total_frames-1, 8, dtype=int)
                
                for i in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                cap.release()
            
            if not frames:
                raise ValueError("No se pudieron leer frames del video")
            
            # Procesamos los frames
            frames_features = []
            for frame in frames:
                frame_tensor = self.img_transform(frame).unsqueeze(0)
                with torch.no_grad():
                    features = self.cnn3d(frame_tensor.repeat(1, 1, 8, 1, 1).permute(0, 2, 1, 3, 4))
                    frames_features.append(features.flatten())
            
            # Secuencia temporal
            seq_tensor = torch.stack(frames_features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, (hidden, _) = self.lstm(seq_tensor)
                features_temp = hidden[-1].cpu().numpy()
            
            return self._binarizar_features(features_temp)
        except Exception as e:
            raise ValueError(f"Error procesando video: {str(e)}")
    
    def _procesar_texto(self, texto):
        try:
            inputs = self.tokenizer(texto, return_tensors='pt', 
                                  truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.transformer(**inputs)
                features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            return self._binarizar_features(features)
        except Exception as e:
            raise ValueError(f"Error procesando texto: {str(e)}")
    
    def _procesar_audio(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, duration=3.0)
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            plt.figure(figsize=(3, 3))
            librosa.display.specshow(S_dB, sr=sr)
            plt.axis('off')
            plt.savefig('temp_spectrogram.png', bbox_inches='tight', pad_inches=0)
            plt.close()
            
            return self._procesar_imagen('temp_spectrogram.png')
        except Exception as e:
            raise ValueError(f"Error procesando audio: {str(e)}")
    
    def _binarizar_features(self, features):
        flat_features = features.flatten()
        binary_str = ''.join(['1' if x > np.median(flat_features) else '0' for x in flat_features])
        return binary_str[:512]

# ==============================================
# Interfaz Gráfica (Versión Corregida)
# ==============================================
class EsenciaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Extractor de Esencia Multimodal - Corregido")
        self.root.geometry("1000x800")
        
        self.extractor = EsenciaExtractor()
        self.cap = None
        self.capturing = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.current_frame = None
        
        self._crear_interfaz()
    
    def _crear_interfaz(self):
        # Panel de control
        control_frame = ttk.LabelFrame(self.root, text="Opciones", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Tipo de entrada
        ttk.Label(control_frame, text="Tipo de entrada:").pack(anchor='w')
        self.tipo_var = tk.StringVar(value='imagen')
        tipos = ['imagen', 'video', 'texto', 'audio']
        for tipo in tipos:
            ttk.Radiobutton(control_frame, text=tipo.capitalize(), 
                           variable=self.tipo_var, value=tipo,
                           command=self._actualizar_controles).pack(anchor='w')
        
        # Modo de entrada
        ttk.Label(control_frame, text="Fuente:").pack(anchor='w', pady=(10, 0))
        self.fuente_var = tk.StringVar(value='archivo')
        ttk.Radiobutton(control_frame, text="Importar archivo", 
                       variable=self.fuente_var, value='archivo',
                       command=self._actualizar_controles).pack(anchor='w')
        ttk.Radiobutton(control_frame, text="Cámara web", 
                       variable=self.fuente_var, value='camara',
                       command=self._actualizar_controles).pack(anchor='w')
        
        # Botones de acción
        self.seleccionar_btn = ttk.Button(control_frame, text="Seleccionar archivo", 
                                        command=self.seleccionar_archivo)
        self.seleccionar_btn.pack(fill=tk.X, pady=5)
        
        self.camara_btn = ttk.Button(control_frame, text="Iniciar cámara", 
                                    command=self.iniciar_camara)
        self.camara_btn.pack(fill=tk.X, pady=5)
        
        self.capturar_btn = ttk.Button(control_frame, text="Capturar imagen", 
                                      command=self.capturar_imagen, state='disabled')
        self.capturar_btn.pack(fill=tk.X, pady=5)
        
        self.procesar_btn = ttk.Button(control_frame, text="Extraer esencia", 
                                      command=self.extraer_esencia, state='disabled')
        self.procesar_btn.pack(fill=tk.X, pady=5)
        
        self.detener_btn = ttk.Button(control_frame, text="Detener cámara", 
                                     command=self.detener_camara, state='disabled')
        self.detener_btn.pack(fill=tk.X, pady=5)
        
        # Área de visualización
        vis_frame = ttk.Frame(self.root)
        vis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Vista previa
        self.preview_label = ttk.Label(vis_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Resultados
        result_frame = ttk.LabelFrame(vis_frame, text="Resultados", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.result_text = tk.Text(result_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_text.configure(yscrollcommand=scrollbar.set)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Variables de estado
        self.archivo_path = ""
        self._actualizar_controles()
    
    def _actualizar_controles(self):
        tipo = self.tipo_var.get()
        fuente = self.fuente_var.get()
        
        # Habilitar/deshabilitar controles según selección
        self.seleccionar_btn.config(state='normal' if fuente == 'archivo' else 'disabled')
        self.camara_btn.config(state='normal' if fuente == 'camara' and tipo in ['imagen', 'video'] else 'disabled')
        self.procesar_btn.config(state='normal' if (fuente == 'archivo' and self.archivo_path) or 
                                                  (fuente == 'camara' and self.current_frame is not None) else 'disabled')
    
    def seleccionar_archivo(self):
        tipo = self.tipo_var.get()
        filetypes = []
        
        if tipo == 'imagen':
            filetypes = [("Imágenes", "*.jpg *.jpeg *.png")]
        elif tipo == 'video':
            filetypes = [("Videos", "*.mp4 *.avi *.mov")]
        elif tipo == 'audio':
            filetypes = [("Audio", "*.wav *.mp3")]
        elif tipo == 'texto':
            filetypes = [("Archivos de texto", "*.txt")]
        
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.archivo_path = path
            self.mostrar_preview()
            self._actualizar_controles()
    
    def iniciar_camara(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.result_text.insert(tk.END, "Error: No se pudo abrir la cámara\n")
                return
            
            self.capturing = True
            self.capturar_btn.config(state='normal')
            self.detener_btn.config(state='normal')
            self._actualizar_camara()
    
    def _actualizar_camara(self):
        if self.capturing and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Actualizar frame actual
                self.current_frame = frame.copy()
                
                # Mostrar en la interfaz
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img.thumbnail((600, 600))
                imgtk = ImageTk.PhotoImage(image=img)
                self.preview_label.imgtk = imgtk
                self.preview_label.configure(image=imgtk)
            
            self.root.after(30, self._actualizar_camara)
    
    def capturar_imagen(self):
        if self.current_frame is not None:
            self.capturing = False
            self.procesar_btn.config(state='normal')
            self.result_text.insert(tk.END, "Imagen capturada lista para procesar\n")
    
    def detener_camara(self):
        self.capturing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.capturar_btn.config(state='disabled')
        self.detener_btn.config(state='disabled')
        self.current_frame = None
        self.preview_label.configure(image='')
        self._actualizar_controles()
    
    def mostrar_preview(self):
        tipo = self.tipo_var.get()
        fuente = self.fuente_var.get()
        
        try:
            if fuente == 'archivo':
                if tipo == 'imagen':
                    img = Image.open(self.archivo_path)
                    img.thumbnail((600, 600))
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.preview_label.imgtk = imgtk
                    self.preview_label.configure(image=imgtk)
                
                elif tipo == 'video':
                    cap = cv2.VideoCapture(self.archivo_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame)
                        img.thumbnail((600, 600))
                        imgtk = ImageTk.PhotoImage(image=img)
                        self.preview_label.imgtk = imgtk
                        self.preview_label.configure(image=imgtk)
                
                elif tipo == 'texto':
                    with open(self.archivo_path, 'r', encoding='utf-8') as f:
                        contenido = f.read(200) + "..." if len(f.read()) > 200 else f.read()
                    self.preview_label.configure(text=f"Texto a procesar:\n{contenido}", image='')
                
                elif tipo == 'audio':
                    self.preview_label.configure(text=f"Archivo de audio:\n{os.path.basename(self.archivo_path)}", image='')
            
            self.procesar_btn.config(state='normal')
        
        except Exception as e:
            self.result_text.insert(tk.END, f"Error al mostrar vista previa: {str(e)}\n")
    
    def extraer_esencia(self):
        tipo = self.tipo_var.get()
        fuente = self.fuente_var.get()
        
        try:
            input_data = None
            if fuente == 'archivo':
                if not self.archivo_path:
                    raise ValueError("No se ha seleccionado ningún archivo")
                
                if tipo == 'texto':
                    with open(self.archivo_path, 'r', encoding='utf-8') as f:
                        input_data = f.read()
                else:
                    input_data = self.archivo_path
            
            elif fuente == 'camara':
                if tipo == 'imagen' and self.current_frame is not None:
                    input_data = self.current_frame
                else:
                    raise ValueError("No hay datos capturados de la cámara")
            
            if input_data is None:
                raise ValueError("No hay datos para procesar")
            
            # Deshabilitar botón durante el procesamiento
            self.procesar_btn.config(state='disabled')
            self.result_text.insert(tk.END, f"\nIniciando procesamiento de {tipo}...\n")
            
            # Ejecutar en un hilo separado
            threading.Thread(
                target=self._procesar_en_hilo,
                args=(input_data, tipo),
                daemon=True
            ).start()
        
        except Exception as e:
            self.result_text.insert(tk.END, f"\nError al iniciar procesamiento: {str(e)}\n")
            self.procesar_btn.config(state='normal')
    
    def _procesar_en_hilo(self, input_data, tipo):
        try:
            esencia = self.extractor.extraer_esencia(input_data, tipo)
            
            self.root.after(0, lambda: self._mostrar_resultados(esencia, tipo))
        
        except Exception as e:
            self.root.after(0, lambda: self.result_text.insert(
                tk.END, f"\nError durante el procesamiento: {str(e)}\n"))
        finally:
            self.root.after(0, lambda: self.procesar_btn.config(state='normal'))
    
    def _mostrar_resultados(self, esencia, tipo):
        self.result_text.insert(tk.END, f"\nEsencia extraída ({tipo}):\n")
        self.result_text.insert(tk.END, f"Longitud: {len(esencia)} bits\n")
        self.result_text.insert(tk.END, f"Primeros 100 bits:\n{esencia[:100]}...\n\n")
        
        # Visualizar esencia
        self._visualizar_esencia(esencia, tipo)
    
    def _visualizar_esencia(self, bin_str, tipo):
        bits = np.array([int(b) for b in bin_str])
        side = int(np.sqrt(len(bits))) + 1
        bits = np.pad(bits, (0, side*side - len(bits)))
        img = bits.reshape((side, side))
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='binary')
        plt.title(f"Esencia de {tipo}")
        plt.axis('off')
        
        # Guardar temporalmente
        temp_path = "esencia_temp.png"
        plt.savefig(temp_path, bbox_inches='tight')
        plt.close()
        
        # Mostrar en interfaz
        img_esencia = Image.open(temp_path)
        img_esencia.thumbnail((400, 400))
        imgtk = ImageTk.PhotoImage(image=img_esencia)
        
        # Crear ventana para visualización
        esencia_window = tk.Toplevel(self.root)
        esencia_window.title("Visualización de Esencia")
        
        label = ttk.Label(esencia_window, image=imgtk)
        label.image = imgtk
        label.pack(padx=10, pady=10)
        
        # Botón para guardar
        ttk.Button(
            esencia_window, 
            text="Guardar Esencia", 
            command=lambda: self._guardar_esencia(bin_str, tipo)
        ).pack(pady=5)
    
    def _guardar_esencia(self, esencia, tipo):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivo de texto", "*.txt")],
            title=f"Guardar esencia de {tipo}"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(esencia)
            self.result_text.insert(tk.END, f"\nEsencia guardada en: {file_path}\n")

# ==============================================
# Ejecución
# ==============================================
if __name__ == "__main__":
    root = tk.Tk()
    app = EsenciaApp(root)
    root.mainloop()