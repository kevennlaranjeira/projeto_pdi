# processamento.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter, maximum_filter

# --- Parte III: Análise e Transformações de Intensidade ---

def calcular_e_exibir_histograma(imagem):
    """Calcula e exibe o histograma de uma imagem em tons de cinza."""
    if imagem is None:
        print("Imagem de entrada inválida para o histograma.")
        return
    # Parâmetros do calcHist preenchidos corretamente:
    # canais=[0], mask=None, bins=256, range 0-256
    hist = cv2.calcHist([imagem], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Histograma da Imagem")
    plt.xlabel("Intensidade")
    plt.ylabel("Número de pixels")
    plt.plot(hist)  # ou plt.bar se preferir
    plt.xlim([0, 256])
    plt.show()

def alargar_contraste(imagem):
    """Realça contraste esticando valores de pixel para toda faixa [0,255]."""
    if imagem is None:
        return None
    min_val = np.min(imagem)
    max_val = np.max(imagem)
    if max_val - min_val == 0:
        return imagem.copy()
    stretched = ((imagem - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
    return stretched

def equalizar_histograma(imagem):
    """Equaliza histograma de imagem em tons de cinza."""
    if imagem is None:
        return None
    return cv2.equalizeHist(imagem)

def aplicar_filtro_media(imagem, tamanho_kernel):
    return cv2.blur(imagem, (tamanho_kernel, tamanho_kernel))

def aplicar_filtro_mediana(imagem, tamanho_kernel):
    return cv2.medianBlur(imagem, tamanho_kernel)

def aplicar_filtro_gaussiano(imagem, tamanho_kernel, sigma):
    return cv2.GaussianBlur(imagem, (tamanho_kernel, tamanho_kernel), sigma)

def aplicar_filtro_minimo(imagem, tamanho_kernel):
    return minimum_filter(imagem, size=tamanho_kernel)

def aplicar_filtro_maximo(imagem, tamanho_kernel):
    return maximum_filter(imagem, size=tamanho_kernel)

def aplicar_filtro_laplaciano(imagem):
    """Aplica Laplaciano para detecção de bordas."""
    if imagem is None:
        return None
    return cv2.Laplacian(imagem, cv2.CV_64F).astype(np.uint8)

def aplicar_filtro_roberts(imagem):
    """Operador Roberts (2x2)."""
    if imagem is None:
        return None
    # Definição simples de Roberts:
    kernel_x = np.array([[1, 0], [0, -1]], dtype=int)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=int)
    grad_x = cv2.filter2D(imagem, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(imagem, cv2.CV_64F, kernel_y)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    return cv2.convertScaleAbs(grad)

def aplicar_filtro_prewitt(imagem):
    """Aplica o operador de Prewitt para detecção de bordas."""
    if imagem is None:
        return None
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=int)
    # Kernel Y de Prewitt definido corretamente:
    kernel_y = np.array([[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]], dtype=int)
    x = cv2.filter2D(imagem, cv2.CV_64F, kernel_x)
    y = cv2.filter2D(imagem, cv2.CV_64F, kernel_y)
    gradiente = np.sqrt(x**2 + y**2)
    return cv2.convertScaleAbs(gradiente)

def aplicar_filtro_sobel(imagem):
    """Aplica o operador Sobel."""
    if imagem is None:
        return None
    grad_x = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    return cv2.convertScaleAbs(grad)

def calcular_e_exibir_espectro(imagem):
    """Calcula e exibe o espectro de frequência (magnitude)."""
    if imagem is None:
        print("Imagem de entrada inválida para espectro.")
        return
    # converter para float e centrar
    f = np.fft.fft2(imagem)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    plt.figure()
    plt.title("Espectro de Magnitude")
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.show()

def aplicar_filtro_frequencia(imagem, raio_corte, tipo_filtro):
    """Aplica filtro passa-baixa ou passa-alta ideal."""
    if imagem is None:
        return None
    rows, cols = imagem.shape
    crow, ccol = rows//2, cols//2
    # FFT
    f = np.fft.fft2(imagem)
    fshift = np.fft.fftshift(f)
    # máscara
    mask = np.zeros_like(imagem, dtype=np.uint8)
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    if tipo_filtro == 'passa-baixa':
        mask[dist <= raio_corte] = 1
    else:  # 'passa-alta'
        mask[dist > raio_corte] = 1
    # aplicar máscara
    fshift_filtered = fshift * mask
    # inversa
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.uint8(np.clip(img_back, 0, 255))
    return img_back

# --- Parte VI: Morfologia Matemática e Segmentação ---

def aplicar_erosao(imagem, tamanho_kernel, formato_kernel):
    kernel = cv2.getStructuringElement(formato_kernel, (tamanho_kernel, tamanho_kernel))
    return cv2.erode(imagem, kernel, iterations=1)

def aplicar_dilatacao(imagem, tamanho_kernel, formato_kernel):
    kernel = cv2.getStructuringElement(formato_kernel, (tamanho_kernel, tamanho_kernel))
    return cv2.dilate(imagem, kernel, iterations=1)

def aplicar_segmentacao_otsu(imagem):
    if imagem is None:
        return None, None
    # garante tons de cinza suavizados
    imagem_suavizada = cv2.GaussianBlur(imagem, (5, 5), 0)
    limiar, imagem_otsu = cv2.threshold(imagem_suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return limiar, imagem_otsu
