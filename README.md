# Classificação de Lesões de Pele com Redes Neurais Convolucionais

## Identificação

- **Disciplina:** Aprendizado Profundo
- **Modelo implementado:** Rede Neural Convolucional (CNN) — implementada do zero em PyTorch
- **Dataset:** Balanced Skin Cancer MNIST: HAM10000 Augmented
- **Link do dataset:** https://www.kaggle.com/datasets/utkarshps/skin-cancer-mnist10000-ham-augmented-dataset
- **Plataforma de execução:** Kaggle Notebooks (GPU habilitada)

---

## Descrição do Problema

Este projeto investiga o uso de uma CNN para a **classificação automática de lesões de pele** em imagens dermatoscópicas, um problema com alto impacto em saúde pública. O diagnóstico precoce de lesões malignas como melanoma e carcinoma basocelular é determinante para a sobrevida dos pacientes, especialmente em regiões com acesso limitado a dermatologistas.

A tarefa é de **classificação multiclasse supervisionada**: dada uma imagem dermatoscópica, o modelo deve identificar qual das sete categorias de lesão ela pertence.

---

## Dataset

| Atributo | Descrição |
|---|---|
| Nome | Balanced Skin Cancer MNIST: HAM10000 Augmented |
| Total de imagens | ~39.507 |
| Número de classes | 7 |
| Formato | JPEG, RGB |
| Resolução (após pré-processamento) | 224 × 224 pixels |

### Classes

| Código | Descrição |
|---|---|
| `akiec` | Ceratose Actínica / Carcinoma de Bowen |
| `bcc` | Carcinoma Basocelular |
| `bkl` | Ceratose Benigna |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Nevo Melanocítico |
| `vasc` | Lesão Vascular |

### Divisão dos dados

| Split | Amostras | Estratégia |
|---|---|---|
| Treino | 32.783 | 85% do `train_dir` — divisão estratificada |
| Teste | 5.786 | 15% do `train_dir` — divisão estratificada |
| Validação | 938 | `val_dir` fornecida pelo dataset |

A divisão foi feita com `train_test_split` do scikit-learn usando `stratify=labels` e `random_state=42`, garantindo que cada classe perca exatamente 15% das amostras para o teste.

---

## Arquitetura do Modelo

A `RedeCNN` é composta por dois blocos principais:

```
Entrada (3 × 224 × 224)
    │
    ▼
Conv2d(3→32, kernel=3, padding=1) → ReLU → MaxPool2d(2×2)   # saída: 32 × 112 × 112
    │
    ▼
Conv2d(32→64, kernel=3, padding=1) → ReLU → MaxPool2d(2×2)  # saída: 64 × 56 × 56
    │
    ▼
Conv2d(64→128, kernel=3, padding=1) → ReLU → MaxPool2d(2×2) # saída: 128 × 28 × 28
    │
    ▼
Flatten → Linear(100352, 512) → ReLU → Dropout(0.5) → Linear(512, 7)
    │
    ▼
Saída (7 classes)
```

---

## Ambiente

- **Linguagem:** Python 3.12.13
- **Plataforma:** Kaggle Notebooks
- **GPU:** CUDA (habilitada via `torch.device('cuda')`)
- **Principais bibliotecas:**

```
torch
torchvision
scikit-learn
matplotlib
pandas
tqdm
kagglehub
```

---

## Hiperparâmetros

| Hiperparâmetro | Valor |
|---|---|
| Otimizador | Adam |
| Taxa de aprendizado | 1e-3 |
| Batch size (treino) | 200 |
| Batch size (validação) | 28 |
| Batch size (teste) | 64 |
| Função de perda | CrossEntropyLoss |
| Dropout | 0,5 |
| Semente aleatória | 42 |
| `num_workers` | 4 (treino) / 3 (val e teste) |
| `persistent_workers` | True |

---

## Como Reproduzir

### 1. No Kaggle (recomendado)

1. Acesse [kaggle.com](https://www.kaggle.com) e crie uma conta, se ainda não tiver
2. Crie um novo **Notebook**
3. Adicione o dataset: clique em **Add Input** → busque por `skin-cancer-mnist10000-ham-augmented-dataset` → adicione
4. Habilite a GPU: **Settings → Accelerator → GPU**
5. Copie o conteúdo do arquivo `nota-3.ipynb` para o notebook
6. Execute todas as células em ordem (**Run All**)

O path do dataset será resolvido automaticamente pelo `kagglehub`:

```python
import kagglehub
DATASET = kagglehub.dataset_download("utkarshps/skin-cancer-mnist10000-ham-augmented-dataset")
```
