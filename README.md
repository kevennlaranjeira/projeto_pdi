# 🎨 Sistema Interativo de Processamento de Imagens - SIN 392

Este repositório contém o projeto prático desenvolvido para a disciplina **SIN 392 - Introdução ao Processamento Digital de Imagens**, da **Universidade Federal de Viçosa (UFV)**.

📌 **Objetivo:** Criar um sistema desktop com interface gráfica intuitiva para aplicar técnicas de análise e processamento de imagens, abrangendo todos os conceitos estudados na disciplina.

---

## ✨ Funcionalidades

O sistema permite **carregar uma imagem (colorida ou em tons de cinza)**, processá-la e **salvar o resultado**. Caso a imagem seja colorida, ela é convertida automaticamente para tons de cinza antes de aplicar os filtros.

### 🔹 Análise de Histograma
- Cálculo e exibição do histograma da imagem.

### 🔹 Transformações de Intensidade
- Alargamento de contraste (normalização)  
- Equalização de histograma

### 🔹 Filtros Espaciais
**Passa-baixa:**  
- Média  
- Mediana  
- Gaussiano  
- Mínimo  
- Máximo

**Passa-alta:**  
- Laplaciano  
- Roberts  
- Prewitt  
- Sobel

### 🔹 Domínio da Frequência
- Visualização do espectro de Fourier  
- Filtragem passa-baixa e passa-alta ideal com raio ajustável

### 🔹 Morfologia Matemática
- Erosão  
- Dilatação  
Com elemento estruturante e tamanho de kernel configuráveis.

### 🔹 Segmentação de Imagens
- Limiarização automática utilizando o método de **Otsu**

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3  
- **Interface Gráfica (GUI):** CustomTkinter  
- **Processamento de Imagem:** OpenCV, SciPy  
- **Manipulação Numérica:** NumPy  
- **Gráficos:** Matplotlib  
- **Imagens na GUI:** Pillow (PIL)

---

## 🚀 Como Executar o Projeto

### ✅ Pré-requisitos

- Python 3.8 ou superior  
- Git instalado

### 🔧 Passo a passo

1. **Clone o repositório**

```bash
git clone https://github.com/kevennlaranjeira/projeto_pdi.git
cd projeto_pdi
```

2. **Crie e ative um ambiente virtual**

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Instale as dependências**

```bash
pip install -r requirements.txt
```

4. **Execute a aplicação**

```bash
python main.py
```

---

## 👤 Autor

**Keven Laranjeira**  
[GitHub: kevennlaranjeira](https://github.com/kevennlaranjeira)
