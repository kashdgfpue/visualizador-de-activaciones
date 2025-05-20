import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image, ImageOps
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import json
import os

class AdvancedTensorVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Advanced Tensor Visualizer")
        self.root.geometry("1200x900")
        
        # Inicializar configuración primero
        self.settings = {
            "cmap": "viridis",
            "theme": "light",
            "filter_view": "grid",
            "default_size": 64,
            "show_toolbar": True
        }
        self.load_settings()  # Cargar configuración si existe
        
        # Ahora podemos llamar a setup_styles
        self.setup_styles()
        
        # Modelo CNN
        self.model = self.build_model()
        
        # Interfaz
        self.create_main_interface()
        self.create_settings_panel()
        
    def setup_styles(self):
        """Configura los estilos visuales"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Colores según tema
        self.bg_color = "#f5f5f5" if self.settings["theme"] == "light" else "#2d3436"
        self.fg_color = "#2d3436" if self.settings["theme"] == "light" else "#f5f5f5"
        self.accent_color = "#0984e3"
        
        self.style.configure('.', background=self.bg_color, foreground=self.fg_color)
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, foreground=self.fg_color)
        self.style.configure('TButton', background=self.accent_color, foreground='white')
        self.style.configure('TCombobox', fieldbackground='white')
        self.style.configure('TNotebook', background=self.bg_color)
        self.style.configure('TNotebook.Tab', background=self.bg_color, padding=[10, 5])
        
    def build_model(self):
        """Construye el modelo CNN 3D"""
        class CNN3D(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
                self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
                self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
                self.conv4 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                x = torch.relu(self.conv4(x))
                return x
                
        return CNN3D()
    
    def create_main_interface(self):
        """Crea la interfaz principal"""
        # Panel principal
        self.main_panel = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_panel.pack(fill=tk.BOTH, expand=True)
        
        # Panel de control
        self.control_frame = ttk.Frame(self.main_panel, width=300, padding=10)
        self.main_panel.add(self.control_frame, weight=0)
        
        # Panel de visualización
        self.vis_frame = ttk.Frame(self.main_panel, padding=10)
        self.main_panel.add(self.vis_frame, weight=1)
        
        # Controles principales
        self.create_control_panel()
        
    def create_control_panel(self):
        """Crea el panel de control principal"""
        # Título
        title = ttk.Label(
            self.control_frame, 
            text="Controles", 
            font=('Helvetica', 14, 'bold'),
            foreground=self.accent_color
        )
        title.pack(pady=10)
        
        # Selector de capa
        layer_frame = ttk.LabelFrame(self.control_frame, text="Capa CNN", padding=10)
        layer_frame.pack(fill=tk.X, pady=5)
        
        self.layer_var = tk.StringVar()
        layers = [
            ('Capa 1 - 8 filtros', 'conv1'),
            ('Capa 2 - 16 filtros', 'conv2'),
            ('Capa 3 - 32 filtros', 'conv3'),
            ('Capa 4 - 64 filtros', 'conv4')
        ]
        
        for text, mode in layers:
            rb = ttk.Radiobutton(
                layer_frame, 
                text=text, 
                variable=self.layer_var, 
                value=mode,
                command=self.update_visualization
            )
            rb.pack(anchor=tk.W)
        
        self.layer_var.set('conv1')
        
        # Control de filtro
        filter_frame = ttk.LabelFrame(self.control_frame, text="Filtro", padding=10)
        filter_frame.pack(fill=tk.X, pady=5)
        
        self.filter_slider = ttk.Scale(
            filter_frame,
            from_=0,
            to=7,
            orient=tk.HORIZONTAL,
            command=lambda e: self.update_visualization()
        )
        self.filter_slider.pack(fill=tk.X)
        
        self.filter_label = ttk.Label(filter_frame, text="Filtro: 0")
        self.filter_label.pack()
        
        # Visualización
        view_frame = ttk.LabelFrame(self.control_frame, text="Visualización", padding=10)
        view_frame.pack(fill=tk.X, pady=5)
        
        self.view_mode = tk.StringVar(value='grid')
        ttk.Radiobutton(
            view_frame, 
            text="Cuadrícula", 
            variable=self.view_mode, 
            value='grid',
            command=self.update_visualization
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            view_frame, 
            text="Individual", 
            variable=self.view_mode, 
            value='single',
            command=self.update_visualization
        ).pack(anchor=tk.W)
        
        # Botones de acción
        btn_frame = ttk.Frame(self.control_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            btn_frame, 
            text="Cargar Imagen", 
            command=self.load_image
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        ttk.Button(
            btn_frame, 
            text="Ajustes", 
            command=self.toggle_settings
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        ttk.Button(
            btn_frame, 
            text="Guardar", 
            command=self.save_visualization
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
    def create_settings_panel(self):
        """Crea el panel de configuración (oculto inicialmente)"""
        self.settings_panel = ttk.Frame(self.root, padding=10)
        
        # Configuración de visualización
        settings_frame = ttk.LabelFrame(self.settings_panel, text="Configuración", padding=10)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Selector de colormap
        ttk.Label(settings_frame, text="Mapa de colores:").pack(anchor=tk.W)
        self.cmap_var = tk.StringVar(value=self.settings["cmap"])
        cmaps = ['viridis', 'plasma', 'magma', 'inferno', 'cividis', 'coolwarm']
        cmap_menu = ttk.Combobox(
            settings_frame, 
            textvariable=self.cmap_var, 
            values=cmaps,
            state='readonly'
        )
        cmap_menu.pack(fill=tk.X, pady=5)
        
        # Tema
        ttk.Label(settings_frame, text="Tema:").pack(anchor=tk.W)
        self.theme_var = tk.StringVar(value=self.settings["theme"])
        ttk.Radiobutton(
            settings_frame, 
            text="Claro", 
            variable=self.theme_var, 
            value='light'
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            settings_frame, 
            text="Oscuro", 
            variable=self.theme_var, 
            value='dark'
        ).pack(anchor=tk.W)
        
        # Barra de herramientas
        self.toolbar_var = tk.BooleanVar(value=self.settings["show_toolbar"])
        ttk.Checkbutton(
            settings_frame,
            text="Mostrar barra de herramientas",
            variable=self.toolbar_var
        ).pack(anchor=tk.W, pady=5)
        
        # Botones de configuración
        btn_frame = ttk.Frame(settings_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            btn_frame,
            text="Aplicar",
            command=self.apply_settings
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        ttk.Button(
            btn_frame,
            text="Cancelar",
            command=self.toggle_settings
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
    def toggle_settings(self):
        """Muestra/oculta el panel de configuración"""
        if self.settings_panel.winfo_ismapped():
            self.settings_panel.pack_forget()
        else:
            self.settings_panel.pack(fill=tk.BOTH, expand=True)
    
    def apply_settings(self):
        """Aplica la configuración seleccionada"""
        self.settings = {
            "cmap": self.cmap_var.get(),
            "theme": self.theme_var.get(),
            "show_toolbar": self.toolbar_var.get()
        }
        self.setup_styles()
        self.update_visualization()
        self.save_settings()
        self.toggle_settings()
    
    def save_settings(self):
        """Guarda la configuración en un archivo JSON"""
        try:
            with open('tensor_vis_settings.json', 'w') as f:
                json.dump(self.settings, f)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la configuración: {str(e)}")
    
    def load_settings(self):
        """Carga la configuración desde un archivo JSON"""
        try:
            if os.path.exists('tensor_vis_settings.json'):
                with open('tensor_vis_settings.json', 'r') as f:
                    self.settings.update(json.load(f))
        except Exception as e:
            messagebox.showwarning("Advertencia", f"No se pudo cargar la configuración: {str(e)}")
    
    def update_visualization(self):
        """Actualiza la visualización según los parámetros seleccionados"""
        # Limpiar el frame de visualización
        for widget in self.vis_frame.winfo_children():
            widget.destroy()
        
        # Obtener capa seleccionada
        layer_name = self.layer_var.get()
        layer = getattr(self.model, layer_name)
        filters = layer.weight.detach().cpu()
        
        # Actualizar slider según número de filtros
        num_filters = filters.shape[0]
        self.filter_slider.config(to=num_filters-1)
        current_filter = int(self.filter_slider.get())
        self.filter_label.config(text=f"Filtro: {current_filter+1}/{num_filters}")
        
        # Crear visualización según modo
        if self.view_mode.get() == 'grid':
            self.show_filter_grid(filters)
        else:
            self.show_single_filter(filters, current_filter)
    
    def show_filter_grid(self, filters):
        """Muestra todos los filtros en una cuadrícula"""
        num_filters = filters.shape[0]
        cols = 4
        rows = int(np.ceil(num_filters / cols))
        
        fig = plt.figure(figsize=(12, 3*rows))
        
        for i in range(num_filters):
            ax = fig.add_subplot(rows, cols, i+1)
            img = self.tensor_to_image(filters[i, 0], colormap=self.settings["cmap"])
            ax.imshow(img)
            ax.set_title(f'Filtro {i+1}', fontsize=8)
            ax.axis('off')
        
        plt.tight_layout()
        self.display_figure(fig)
    
    def show_single_filter(self, filters, filter_idx):
        """Muestra un único filtro con detalle"""
        fig = plt.figure(figsize=(10, 8))
        
        # Vista principal
        ax1 = fig.add_subplot(2, 1, 1)
        selected_filter = filters[filter_idx, 0]
        img = self.tensor_to_image(selected_filter, colormap=self.settings["cmap"])
        ax1.imshow(img)
        ax1.set_title(f'Filtro {filter_idx+1}', pad=20)
        ax1.axis('off')
        
        # Histograma de valores
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.hist(selected_filter.numpy().flatten(), bins=50, color=self.accent_color)
        ax2.set_title('Distribución de valores')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.display_figure(fig)
    
    def display_figure(self, fig):
        """Muestra una figura matplotlib en el frame"""
        canvas = FigureCanvasTkAgg(fig, master=self.vis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        if self.settings["show_toolbar"]:
            toolbar = NavigationToolbar2Tk(canvas, self.vis_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def tensor_to_image(self, tensor, colormap='viridis'):
        """Convierte un tensor a imagen PIL"""
        img_np = tensor.numpy()
        
        # Normalización
        norm = Normalize(vmin=img_np.min(), vmax=img_np.max())
        img_np = norm(img_np)
        
        # Aplicar colormap
        cmap = plt.get_cmap(colormap)
        img_np = cmap(img_np)[..., :3]  # Ignorar canal alpha
        
        # Convertir a uint8
        img_np = (img_np * 255).astype(np.uint8)
        
        return Image.fromarray(img_np, 'RGB')
    
    def load_image(self):
        """Carga una imagen para visualizar activaciones"""
        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=(("Imágenes", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*"))
        )
        
        if filepath:
            try:
                # Procesar imagen
                img = Image.open(filepath)
                img_tensor = self.preprocess_image(img)
                activations = self.get_activations(img_tensor)
                
                # Mostrar en ventana nueva
                self.show_activations_window(img, activations)
                
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo procesar la imagen: {str(e)}")
    
    def preprocess_image(self, img):
        """Preprocesa una imagen para la red"""
        img = img.convert('L')  # Escala de grises
        img = ImageOps.fit(img, (self.settings["default_size"], self.settings["default_size"]))
        
        # Convertir a tensor 3D
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        img_tensor = img_tensor.unsqueeze(2).repeat(1, 1, 16, 1, 1)  # 1x1x16xHxW
        
        return img_tensor
    
    def get_activations(self, x):
        """Obtiene activaciones para todas las capas"""
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # Registrar hooks
        hooks = []
        for name, layer in self.model.named_children():
            if isinstance(layer, nn.Conv3d):
                hooks.append(layer.register_forward_hook(get_activation(name)))
        
        # Forward pass
        with torch.no_grad():
            self.model(x)
        
        # Eliminar hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def show_activations_window(self, original_img, activations):
        """Muestra las activaciones en una nueva ventana"""
        window = tk.Toplevel(self.root)
        window.title("Activaciones CNN")
        window.geometry("1200x800")
        
        notebook = ttk.Notebook(window)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de imagen original
        orig_frame = ttk.Frame(notebook)
        notebook.add(orig_frame, text="Imagen Original")
        
        fig_orig = plt.figure(figsize=(6, 6))
        plt.imshow(original_img, cmap='gray')
        plt.title("Imagen de Entrada")
        plt.axis('off')
        plt.tight_layout()
        
        canvas_orig = FigureCanvasTkAgg(fig_orig, master=orig_frame)
        canvas_orig.draw()
        canvas_orig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Pestaña para cada capa
        for layer_name, act in activations.items():
            layer_frame = ttk.Frame(notebook)
            notebook.add(layer_frame, text=layer_name)
            
            fig = plt.figure(figsize=(10, 8))
            self.plot_activations(fig, act, layer_name)
            
            canvas = FigureCanvasTkAgg(fig, master=layer_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            if self.settings["show_toolbar"]:
                toolbar = NavigationToolbar2Tk(canvas, layer_frame)
                toolbar.update()
    
    def plot_activations(self, fig, activations, layer_name):
        """Grafica las activaciones de una capa"""
        num_filters = activations.shape[1]
        cols = 4
        rows = int(np.ceil(num_filters / cols))
        
        for i in range(num_filters):
            ax = fig.add_subplot(rows, cols, i+1)
            
            # Tomar slice central en la dimensión Z
            z_slice = activations.shape[2] // 2
            activation_slice = activations[0, i, z_slice].cpu().numpy()
            
            ax.imshow(activation_slice, cmap=self.settings["cmap"])
            ax.set_title(f'Filtro {i+1}', fontsize=8)
            ax.axis('off')
        
        fig.suptitle(f'Activaciones - {layer_name}', y=1.02)
        plt.tight_layout()
    
    def save_visualization(self):
        """Guarda la visualización actual como imagen"""
        filepath = filedialog.asksaveasfilename(
            title="Guardar visualización",
            defaultextension=".png",
            filetypes=(("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos los archivos", "*.*"))
        )
        
        if filepath:
            try:
                # Obtener figura actual
                for widget in self.vis_frame.winfo_children():
                    if isinstance(widget, FigureCanvasTkAgg):
                        fig = widget.figure
                        fig.savefig(filepath, dpi=300, bbox_inches='tight')
                        messagebox.showinfo("Éxito", f"Visualización guardada en:\n{filepath}")
                        break
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedTensorVisualizer(root)
    root.mainloop()
