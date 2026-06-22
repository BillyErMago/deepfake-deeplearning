# Deepfake Detection with Custom CNN and ResNet-50

Questo progetto si occupa di **Deepfake Detection**, risolvendo un task di classificazione binaria (REAL vs FAKE) partendo da un dataset scaricato da Kaggle.
Il codice mette a confronto due diverse architetture: una **Custom CNN** costruita interamente da zero e un modello **ResNet-50** pre-addestrato e riadattato.

## Configurazione e Installazione

Per evitare conflitti tra dipendenze, è fortemente raccomandato l'uso di un Virtual Environment Python. Segui questi step dal terminale (nella cartella root del progetto):

1. **Crea l'ambiente virtuale**:
```bash
python -m venv venv
```

2. **Attiva l'ambiente**:
- Su Linux / macOS:
  ```bash
  source venv/bin/activate
  ```
- Su Windows:
  ```bash
  venv\Scripts\activate
  ```

3. **Installa le dipendenze**:
```bash
pip install -r requirements.txt
```

## Struttura del Dataset

Assicurati che il file `FINAL_DATASET.csv` (scaricato da Kaggle) si trovi nella directory root del progetto.
Il primo step è scaricare localmente tutte le immagini dichiarate all'interno del CSV:
```bash
python download_dataset.py
```
*Questo comando leggerà le URL e scaricherà le immagini all'interno della cartella `./data/images/`.*

---

## Addestramento del Modello

Puoi lanciare il training con il backbone che preferisci. Lo script provvederà automaticamente ad assegnare la dimensione 224x224 alle immagini in ingresso.
*(Nota: L'Early Stopping interverrà automaticamente nel caso in cui la rete dovesse bloccarsi).*

**1. Per addestrare la Custom CNN (Default):**
```bash
python main.py   // qui ho usato lr = 0.001, e ho usato batch size = 64
```
*In alternativa: `python main.py --backbone custom`*

**2. Per addestrare la ResNet-50:**
```bash
python main.py --backbone resnet50   //se mi da errore vado sul main e abbasso la batch size finche non mi funziona sempre in potenze di 2  e segnarsi i batch size
```

Se vuoi ridurre o aumentare il numero di epoche (il default è 50) e di batch_size, puoi usare parametri aggiuntivi:
```bash
python main.py --backbone resnet50 --epochs 100 --batch_size 32
```

**Nota**
Se da errore *OUT OF MEMORY* durante l'addestramento, prova a ridurre il `batch_size` a 16 o 8.

---

## Testare il Modello

Per valutare il modello sul set di test e generare le metriche finali, basta lanciare lo script col flag `--test`. Lo script andrà a pescare automaticamente i pesi salvati nella cartella `./checkpoints/`.

**1. Test sulla Custom CNN:**
```bash
python main.py --backbone custom --test
```

**2. Test sulla ResNet-50:**
```bash
python main.py --backbone resnet50 --test
```

## Plottare i Risultati

Terminato l'addestramento e il testing, puoi facilmente stampare i grafici di evoluzione della Loss/Accuracy e le relative Matrici di Confusione:
```bash
python plot.py
```
Le immagini (.png) e i report testuali generati verranno salvati direttamente nelle relative sottocartelle dentro `checkpoints/`.
