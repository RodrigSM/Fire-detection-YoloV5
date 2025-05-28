# 🔥 Detecção de Incêndios com YOLOv5

<p align="center">
  <img src="results/result.gif" alt="Exemplo de Detecção de Incêndio" width="600"/>
</p>

## 📝 Resumo

Este projeto apresenta uma solução baseada em Deep Learning para a deteção automática de incêndios em imagens e vídeos, utilizando o modelo YOLOv5. O sistema é capaz de identificar focos de fogo em tempo real, sendo aplicável em cenários de monitorização florestal, industrial, urbano, entre outros.

## 💡 Motivação

Os incêndios representam uma ameaça significativa ao meio ambiente, à vida humana e à infraestrutura. A deteção precoce é fundamental para minimizar danos e salvar vidas. Soluções automáticas baseadas em visão computacional podem acelerar a resposta a emergências e reduzir custos operacionais.

## 🎯 Objetivos

- Detetar focos de incêndio em imagens e vídeos com elevada precisão e rapidez.
- Facilitar a integração do sistema em câmaras de monitorização e drones.
- Disponibilizar um pipeline completo, do treino à inferência, para uso académico e prático.

## ✨ Diferenciais do Projeto

- **Baseado em YOLOv5:** Um dos modelos mais rápidos e precisos para deteção de objetos.
- **Customização fácil:** Permite treinar com diferentes conjuntos de dados e ajustar hiperparâmetros.
- **Pipeline completo:** Inclui scripts para treino, inferência, avaliação e visualização de resultados.
- **Documentação detalhada:** README estruturado para facilitar o uso e a compreensão do projeto.

## ⚙️ Metodologia

1. **Recolha de Dados:** Utilização de datasets públicos e/ou próprios, organizados na pasta `datasets/`.
2. **Anotação:** As imagens são anotadas no formato YOLO (bounding boxes).
3. **Configuração:** O ficheiro `fire.yaml` define as classes e caminhos dos dados.
4. **Treino:** O modelo YOLOv5 é treinado com transfer learning, utilizando pesos pré-treinados.
5. **Avaliação:** Métricas como precisão, recall, F1-score e curvas PR são geradas.
6. **Inferência:** O modelo treinado é utilizado para detetar incêndios em novas imagens e vídeos.

## 🖥️ Requisitos

- Python 3.8+
- PyTorch
- OpenCV
- Dependências do YOLOv5 (ver `yolov5/requirements.txt`)

## 🚀 Como Utilizar

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

- Coloca as tuas imagens e anotações em `datasets/`
- Edita o ficheiro `fire.yaml` para apontar para os teus dados

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

## 🧪 Exemplos de Utilização

- **Monitorização florestal:** Drones ou câmaras fixas a detetar incêndios em tempo real.
- **Ambientes industriais:** Identificação de focos de fogo em fábricas e armazéns.
- **Cidades inteligentes:** Integração com sistemas de segurança urbana.

## 📊 Resultados Esperados

- Deteção rápida e precisa de incêndios em diferentes cenários.
- Redução de falsos positivos com ajuste de hiperparâmetros e inclusão de imagens negativas.
- Facilidade de adaptação para outros tipos de deteção (ex: fumo, explosões).

## 🚧 Limitações e Trabalhos Futuros

- **Falsos positivos:** Luzes vermelhas ou reflexos podem ser confundidos com fogo.
- **Generalização:** O modelo pode necessitar de mais dados para funcionar em ambientes muito diferentes.
- **Trabalhos futuros:** 
  - Adicionar deteção de fumo.
  - Implementar notificação automática (e-mail, SMS).
  - Testar em tempo real com câmaras IP.

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
├── README.md              # Este ficheiro
└── ...                    # Outros ficheiros e scripts
```

## 🧩 Principais Ficheiros

- `yolov5/` - Código original do YOLOv5 (PyTorch)
- `yolov5/detect_fire.py` - Script customizado para deteção de fogo
- `model/yolov5s_best.pt` - Modelo treinado para deteção de incêndio
- `fire.yaml` - Configuração do dataset customizado
- `train.ipynb` - Notebook para experimentação e análise

## 📚 Referências

- [YOLOv5 - Ultralytics](https://github.com/ultralytics/yolov5)
- [Documentação oficial do YOLOv5](https://docs.ultralytics.com/)
- [PyTorch](https://pytorch.org/)
- [Fire Dataset (Kaggle)](https://www.kaggle.com/datasets/atulyakumar98/fire-dataset)
- [Artigo: Real-Time Fire Detection using YOLO](https://arxiv.org/abs/2106.00656)

## 📈 Comparação de Resultados: Dataset Pequeno vs. Dataset Grande

Abaixo, apresento uma comparação entre dois treinamentos realizados:

- **Treinamento 1:** Apenas 100 imagens (exemplo de resultado abaixo)
- **Treinamento 2:** 7.800 imagens (exemplo de resultado abaixo)

### 🔹 Exemplo de Resultados do Último Treino

<p align="center">
  <img src="yolov5/runs/train/exp26/results.png" alt="Resultados de treinamento" width="800"/>
</p>

- **Observações:**  
  - O gráfico acima mostra as principais métricas do treino, como perdas (loss), precisão, recall e mAP.
  - Com mais imagens no dataset, as curvas tendem a ser mais suaves e os resultados mais estáveis.
  - O modelo treinado com mais dados apresenta melhor generalização e desempenho.

### 📊 Conclusão da Comparação

O aumento do número de imagens no dataset resultou num modelo muito mais robusto, fiável e aplicável a situações reais.  
**Quanto maior e mais variado o dataset, melhor o desempenho do modelo de deteção de incêndio!**

## 👨‍💻 Autores

- **Rodrigo Miranda** - r.miranda@ipvc.pt
- **Nelson Freitas** - nelsonfreitas@ipvc.pt

## 📄 Licença

Este projeto está sob a licença MIT. Veja o ficheiro [LICENSE](LICENSE) para mais detalhes.

## 🙏 Agradecimentos

- Professores e colegas da universidade pelo apoio e feedback.
- Comunidade open-source de visão computacional.
- Ultralytics pelo desenvolvimento do YOLOv5.

<p align="center">
  <b>🔥 Deteção de incêndio rápida, eficiente e open-source! 🔥</b>
</p>