# 🔥 Fire Detection with YOLOv5

<p align="center">
  <img src="results/result.gif" alt="Fire Detection Example" width="600"/>
</p>

## 📝 Resumo

Este projeto apresenta uma solução baseada em Deep Learning para detecção automática de incêndios em imagens e vídeos, utilizando o modelo YOLOv5. O sistema é capaz de identificar focos de fogo em tempo real, sendo aplicável em cenários de monitoramento florestal, industrial, urbano, entre outros.

## 💡 Motivação

Incêndios representam uma ameaça significativa ao meio ambiente, à vida humana e à infraestrutura. A detecção precoce é fundamental para minimizar danos e salvar vidas. Soluções automáticas baseadas em visão computacional podem acelerar a resposta a emergências e reduzir custos operacionais.

## 🎯 Objetivos

- Detectar focos de incêndio em imagens e vídeos com alta precisão e rapidez.
- Facilitar a integração do sistema em câmeras de monitoramento e drones.
- Disponibilizar um pipeline completo, do treinamento à inferência, para uso acadêmico e prático.

## ✨ Diferenciais do Projeto

- **Baseado em YOLOv5:** Um dos modelos mais rápidos e precisos para detecção de objetos.
- **Customização fácil:** Permite treinar com diferentes datasets e ajustar hiperparâmetros.
- **Pipeline completo:** Inclui scripts para treinamento, inferência, avaliação e visualização de resultados.
- **Documentação detalhada:** README estruturado para facilitar o uso e a compreensão do projeto.

## ⚙️ Metodologia

1. **Coleta de Dados:** Utilização de datasets públicos e/ou próprios, organizados na pasta `datasets/`.
2. **Anotação:** As imagens são anotadas no formato YOLO (bounding boxes).
3. **Configuração:** Arquivo `fire.yaml` define as classes e caminhos dos dados.
4. **Treinamento:** O modelo YOLOv5 é treinado com transfer learning, utilizando pesos pré-treinados.
5. **Avaliação:** Métricas como precisão, recall, F1-score e curvas PR são geradas.
6. **Inferência:** O modelo treinado é utilizado para detectar incêndios em novas imagens e vídeos.

## 🖥️ Requisitos

- Python 3.8+
- PyTorch
- OpenCV
- Dependências do YOLOv5 (ver `yolov5/requirements.txt`)

## 🚀 Como Usar

### 1. Clonar o repositório

```bash
git clone https://github.com/RodrigSM/Fire-detection-YoloV5.git
cd Fire-detection-YoloV5/yolov5-fire-detection
```

### 2. Instalar dependências

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r yolov5/requirements.txt
```

### 3. Preparar o dataset

- Coloque suas imagens e anotações em `datasets/`
- Edite `fire.yaml` para apontar para seus dados

### 4. Treinar o modelo

```bash
python yolov5/train.py --img 640 --batch 16 --epochs 50 --data fire.yaml --weights yolov5s.pt --project results
```

### 5. Fazer inferência

**Imagem:**
```bash
python yolov5/detect_fire.py --weights model/yolov5s_best.pt --source path/to/image.jpg
```
**Vídeo:**
```bash
python yolov5/detect_fire.py --weights model/yolov5s_best.pt --source input.mp4
```

### 6. Visualizar resultados

- Detecções: pasta `results/`
- Métricas: gráficos em `results/`

## 🧪 Exemplos de Uso

- **Monitoramento florestal:** Drones ou câmeras fixas detectando incêndios em tempo real.
- **Ambientes industriais:** Identificação de focos de fogo em fábricas e depósitos.
- **Cidades inteligentes:** Integração com sistemas de segurança urbana.

## 📊 Resultados Esperados

- Detecção rápida e precisa de incêndios em diferentes cenários.
- Redução de falsos positivos com ajuste de hiperparâmetros e inclusão de imagens negativas.
- Facilidade de adaptação para outros tipos de detecção (ex: fumaça, explosões).

## 🚧 Limitações e Trabalhos Futuros

- **Falsos positivos:** Luzes vermelhas ou reflexos podem ser confundidos com fogo.
- **Generalização:** O modelo pode precisar de mais dados para funcionar em ambientes muito diferentes.
- **Trabalhos futuros:** 
  - Adicionar detecção de fumaça.
  - Implementar notificação automática (e-mail, SMS).
  - Testar em tempo real com câmeras IP.

## 🗂 Estrutura do Projeto

```
yolov5-fire-detection/
│
├── yolov5/                # Código-fonte do YOLOv5
├── datasets/              # (Ignorado pelo git) Base de dados de treino/teste
├── model/                 # Modelos treinados (.pt)
├── results/               # Resultados de inferência e métricas
├── input.mp4              # Exemplo de vídeo de entrada
├── fire.yaml              # Configuração do dataset customizado
├── data.yaml              # Configuração do dataset padrão
├── train.ipynb            # Notebook de treino e avaliação
├── README.md              # Este arquivo
└── ...                    # Outros arquivos e scripts
```

## 🧩 Principais Arquivos

- `yolov5/` - Código original do YOLOv5 (PyTorch)
- `yolov5/detect_fire.py` - Script customizado para detecção de fogo
- `model/yolov5s_best.pt` - Modelo treinado para detecção de incêndio
- `fire.yaml` - Configuração do dataset customizado
- `train.ipynb` - Notebook para experimentação e análise

## 📚 Referências

- [YOLOv5 - Ultralytics](https://github.com/ultralytics/yolov5)
- [Documentação oficial do YOLOv5](https://docs.ultralytics.com/)
- [PyTorch](https://pytorch.org/)
- [Fire Dataset (Kaggle)](https://www.kaggle.com/datasets/atulyakumar98/fire-dataset)
- [Artigo: Real-Time Fire Detection using YOLO](https://arxiv.org/abs/2106.00656)

## 👨‍💻 Autor

- **Seu Nome**
- [Seu LinkedIn](https://www.linkedin.com/)
- [Seu Email]

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🙏 Agradecimentos

- Professores e colegas da universidade pelo apoio e feedback.
- Comunidade open-source de visão computacional.
- Ultralytics pelo desenvolvimento do YOLOv5.

<p align="center">
  <b>🔥 Detecção de incêndio rápida, eficiente e open-source! 🔥</b>
</p>

## 📈 Comparação de Resultados: Dataset Pequeno vs. Dataset Grande

Abaixo, apresento uma comparação entre dois treinamentos realizados:

- **Treinamento 1:** Apenas 100 imagens (`results_19`)
- **Treinamento 2:** 7.800 imagens (`results_26`)

### 🔹 Resultados com 100 imagens

<p align="center">
  <img src="results/results_19_PR_curve.png" alt="PR Curve 100 imagens" width="400"/>
  <img src="results/results_19_P_curve.png" alt="P Curve 100 imagens" width="400"/>
  <img src="results/results_19_R_curve.png" alt="R Curve 100 imagens" width="400"/>
</p>

- **Observações:**  
  - O modelo apresenta overfitting e baixa generalização.  
  - Muitas detecções incorretas (falsos positivos/negativos).  
  - Curvas de precisão e recall instáveis.

---

### 🔹 Resultados com 7.800 imagens

<p align="center">
  <img src="results/results_26_PR_curve.png" alt="PR Curve 7800 imagens" width="400"/>
  <img src="results/results_26_P_curve.png" alt="P Curve 7800 imagens" width="400"/>
  <img src="results/results_26_R_curve.png" alt="R Curve 7800 imagens" width="400"/>
</p>

- **Observações:**  
  - O modelo apresenta alta precisão e recall.  
  - Redução significativa de falsos positivos e negativos.  
  - Curvas de precisão e recall mais suaves e estáveis.  
  - Melhor capacidade de generalização para novos cenários.

---

### 📊 Conclusão da Comparação

O aumento do número de imagens no dataset resultou em um modelo muito mais robusto, confiável e aplicável a situações reais.  
**Quanto maior e mais variado o dataset, melhor o desempenho do modelo de detecção de incêndio!**

---
