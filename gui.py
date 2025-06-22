# gui.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import processamento as proc
import numpy as np
import os

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Configuração da Janela Principal ---
        self.title("SIN 392 - Sistema de Processamento de Imagens")
        self.geometry("1200x700")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Variáveis de estado
        self.original_image = None  # imagem em tons de cinza
        self.processed_image = None

        # Layout principal: frame de controles e frame de exibição
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.display_frame = ctk.CTkFrame(self)
        self.display_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        # Botões de abrir imagem, salvar e reset
        self.btn_open = ctk.CTkButton(self.control_frame, text="Abrir Imagem", command=self.abrir_imagem)
        self.btn_open.pack(pady=5, padx=10, fill="x")
        self.btn_save = ctk.CTkButton(self.control_frame, text="Salvar Imagem", command=self.salvar_imagem)
        self.btn_save.pack(pady=5, padx=10, fill="x")
        self.btn_reset = ctk.CTkButton(self.control_frame, text="Resetar", command=self.resetar)
        self.btn_reset.pack(pady=5, padx=10, fill="x")

        # --- Seção: Spatial Filters ---
        self.spatial_frame = ctk.CTkFrame(self.control_frame)
        self.spatial_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(self.spatial_frame, text="Filtros Espaciais").pack()

        self.label_kernel = ctk.CTkLabel(self.spatial_frame, text="Tamanho do Kernel: 3")
        self.label_kernel.pack(pady=(10,0))
        self.slider_kernel = ctk.CTkSlider(self.spatial_frame, from_=3, to=21, number_of_steps=9,
                                           command=self.atualizar_label_kernel)
        self.slider_kernel.set(3)
        self.slider_kernel.pack(pady=5, padx=10, fill="x")

        self.filtro_espacial_selecionado = ctk.StringVar(value="Média")
        self.menu_filtros = ctk.CTkOptionMenu(self.spatial_frame,
                                              values=["Média", "Mediana", "Gaussiano", "Minimo", "Maximo",
                                                      "Laplaciano", "Roberts", "Prewitt", "Sobel"],
                                              variable=self.filtro_espacial_selecionado)
        self.menu_filtros.pack(pady=5, padx=10, fill="x")

        self.btn_aplicar_filtro = ctk.CTkButton(self.spatial_frame, text="Aplicar Filtro Espacial",
                                               command=self.aplicar_filtro_espacial)
        self.btn_aplicar_filtro.pack(pady=10, padx=10, fill="x")

        # --- Seção: Domínio da Frequência ---
        self.freq_frame = ctk.CTkFrame(self.control_frame)
        self.freq_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(self.freq_frame, text="Domínio da Frequência").pack()

        self.btn_espectro = ctk.CTkButton(self.freq_frame, text="Exibir Espectro", command=self.exibir_espectro)
        self.btn_espectro.pack(pady=5, padx=10, fill="x")

        self.label_raio = ctk.CTkLabel(self.freq_frame, text="Raio de Corte (D0): 30")
        self.label_raio.pack(pady=(10,0))
        self.slider_raio = ctk.CTkSlider(self.freq_frame, from_=1, to=200, number_of_steps=199,
                                         command=self.atualizar_label_raio)
        self.slider_raio.set(30)
        self.slider_raio.pack(pady=5, padx=10, fill="x")

        self.btn_lpf = ctk.CTkButton(self.freq_frame, text="Filtro Passa-Baixa",
                                     command=lambda: self.aplicar_filtro_freq('passa-baixa'))
        self.btn_lpf.pack(pady=5, padx=10, fill="x")
        self.btn_hpf = ctk.CTkButton(self.freq_frame, text="Filtro Passa-Alta",
                                     command=lambda: self.aplicar_filtro_freq('passa-alta'))
        self.btn_hpf.pack(pady=5, padx=10, fill="x")

        # --- Seção: Morfologia Matemática ---
        self.morph_frame = ctk.CTkFrame(self.control_frame)
        self.morph_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(self.morph_frame, text="Morfologia Matemática").pack()

        self.label_kernel_morf = ctk.CTkLabel(self.morph_frame, text="Tamanho do Kernel: 3")
        self.label_kernel_morf.pack(pady=(10,0))
        self.slider_kernel_morf = ctk.CTkSlider(self.morph_frame, from_=3, to=21, number_of_steps=9,
                                                command=self.atualizar_label_kernel_morf)
        self.slider_kernel_morf.set(3)
        self.slider_kernel_morf.pack(pady=5, padx=10, fill="x")

        self.formato_kernel_selecionado = ctk.StringVar(value="Retângulo")
        self.menu_kernel_morf = ctk.CTkOptionMenu(self.morph_frame,
                                                  values=["Retângulo", "Elipse", "Cruz"],
                                                  variable=self.formato_kernel_selecionado)
        self.menu_kernel_morf.pack(pady=5, padx=10, fill="x")

        self.btn_erosao = ctk.CTkButton(self.morph_frame, text="Erosão", command=self.aplicar_morfologia_op_erosao)
        self.btn_erosao.pack(pady=5, padx=10, fill="x")
        self.btn_dilatacao = ctk.CTkButton(self.morph_frame, text="Dilatação", command=self.aplicar_morfologia_op_dilatacao)
        self.btn_dilatacao.pack(pady=5, padx=10, fill="x")

        # --- Seção: Segmentação Otsu ---
        self.seg_frame = ctk.CTkFrame(self.control_frame)
        self.seg_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(self.seg_frame, text="Segmentação").pack()
        self.label_limiar_otsu = ctk.CTkLabel(self.seg_frame, text="Limiar de Otsu: -")
        self.label_limiar_otsu.pack(pady=(10,0))
        self.btn_otsu = ctk.CTkButton(self.seg_frame, text="Aplicar Otsu", command=self.aplicar_otsu)
        self.btn_otsu.pack(pady=5, padx=10, fill="x")

        # Frame de exibição de imagem
        self.canvas = ctk.CTkCanvas(self.display_frame, bg="black")
        self.canvas.pack(expand=True, fill="both")
        self.image_on_canvas = None

    def abrir_imagem(self):
        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Erro", "Não foi possível carregar a imagem.")
            return
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        self.original_image = img_gray
        self.processed_image = img_gray.copy()
        self.exibir_imagem(self.processed_image)

    def salvar_imagem(self):
        if self.processed_image is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem para salvar.")
            return
        # Sugere nome e extensão
        filetypes = [("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")]
        # pega caminho
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=filetypes,
                                            title="Salvar Imagem Como")
        if not path:
            return
        # Determinar formato a partir da extensão
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        # cv2.imwrite detecta formato pelo ext
        success = cv2.imwrite(path, self.processed_image)
        if success:
            messagebox.showinfo("Salvo", f"Imagem salva em:\n{path}")
        else:
            messagebox.showerror("Erro", "Falha ao salvar a imagem.")

    def resetar(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.label_kernel.configure(text="Tamanho do Kernel: 3")
            self.slider_kernel.set(3)
            self.label_raio.configure(text="Raio de Corte (D0): 30")
            self.slider_raio.set(30)
            self.label_kernel_morf.configure(text="Tamanho do Kernel: 3")
            self.slider_kernel_morf.set(3)
            self.label_limiar_otsu.configure(text="Limiar de Otsu: -")
            self.formato_kernel_selecionado.set("Retângulo")
            self.filtro_espacial_selecionado.set("Média")
            self.exibir_imagem(self.processed_image)

    def exibir_imagem(self, imagem):
        """Mostra a imagem (em tons de cinza) no canvas redimensionando para caber."""
        if imagem is None:
            return
        h, w = imagem.shape
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            scale = min(canvas_w/w, canvas_h/h)
            new_w, new_h = int(w*scale), int(h*scale)
        else:
            new_w, new_h = w, h
        img_resized = cv2.resize(imagem, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, image=img_tk, anchor="center")
        self.image_on_canvas = img_tk

    def atualizar_label_kernel(self, value):
        val = int(float(value))
        if val % 2 == 0:
            val += 1
        if val < 3:
            val = 3
        self.slider_kernel.set(val)
        self.label_kernel.configure(text=f"Tamanho do Kernel: {val}")

    def atualizar_label_raio(self, value):
        val = int(float(value))
        if val < 1:
            val = 1
        self.slider_raio.set(val)
        self.label_raio.configure(text=f"Raio de Corte (D0): {val}")

    def atualizar_label_kernel_morf(self, value):
        val = int(float(value))
        if val % 2 == 0:
            val += 1
        if val < 3:
            val = 3
        self.slider_kernel_morf.set(val)
        self.label_kernel_morf.configure(text=f"Tamanho do Kernel: {val}")

    def aplicar_filtro_espacial(self):
        if self.processed_image is None:
            return
        tamanho = int(self.slider_kernel.get())
        escolha = self.filtro_espacial_selecionado.get()
        img = self.processed_image.copy()
        if escolha == "Média":
            img = proc.aplicar_filtro_media(img, tamanho)
        elif escolha == "Mediana":
            img = proc.aplicar_filtro_mediana(img, tamanho)
        elif escolha == "Gaussiano":
            sigma = 0
            img = proc.aplicar_filtro_gaussiano(img, tamanho, sigma)
        elif escolha == "Minimo":
            img = proc.aplicar_filtro_minimo(img, tamanho)
        elif escolha == "Maximo":
            img = proc.aplicar_filtro_maximo(img, tamanho)
        elif escolha == "Laplaciano":
            img = proc.aplicar_filtro_laplaciano(img)
        elif escolha == "Roberts":
            img = proc.aplicar_filtro_roberts(img)
        elif escolha == "Prewitt":
            img = proc.aplicar_filtro_prewitt(img)
        elif escolha == "Sobel":
            img = proc.aplicar_filtro_sobel(img)
        else:
            return
        self.processed_image = img
        self.exibir_imagem(self.processed_image)

    def exibir_espectro(self):
        if self.processed_image is None:
            return
        proc.calcular_e_exibir_espectro(self.processed_image)

    def aplicar_filtro_freq(self, tipo):
        if self.processed_image is None:
            return
        raio = int(self.slider_raio.get())
        img = proc.aplicar_filtro_frequencia(self.processed_image, raio, tipo)
        if img is not None:
            self.processed_image = img
            self.exibir_imagem(self.processed_image)

    def get_formato_kernel(self):
        escolha = self.formato_kernel_selecionado.get()
        if escolha == "Retângulo":
            return cv2.MORPH_RECT
        elif escolha == "Elipse":
            return cv2.MORPH_ELLIPSE
        elif escolha == "Cruz":
            return cv2.MORPH_CROSS
        else:
            return cv2.MORPH_RECT

    def aplicar_morfologia_op_erosao(self):
        if self.processed_image is None:
            return
        tamanho = int(self.slider_kernel_morf.get())
        formato = self.get_formato_kernel()
        img = proc.aplicar_erosao(self.processed_image, tamanho, formato)
        if img is not None:
            self.processed_image = img
            self.exibir_imagem(self.processed_image)

    def aplicar_morfologia_op_dilatacao(self):
        if self.processed_image is None:
            return
        tamanho = int(self.slider_kernel_morf.get())
        formato = self.get_formato_kernel()
        img = proc.aplicar_dilatacao(self.processed_image, tamanho, formato)
        if img is not None:
            self.processed_image = img
            self.exibir_imagem(self.processed_image)

    def aplicar_otsu(self):
        if self.processed_image is not None:
            limiar, imagem_otsu = proc.aplicar_segmentacao_otsu(self.processed_image)
            if imagem_otsu is not None:
                self.processed_image = imagem_otsu
                self.label_limiar_otsu.configure(text=f"Limiar de Otsu: {int(limiar)}")
                self.exibir_imagem(self.processed_image)

if __name__ == "__main__":
    app = App()
    app.mainloop()
