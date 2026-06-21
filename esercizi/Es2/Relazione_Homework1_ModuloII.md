# Homework 1 — End-to-End Reconstruction (Modulo II)
## Relazione completa: richiesta, teoria, implementazione e risultati

---

# 1. Cosa chiedeva il compito

L'Homework 1 del Modulo II chiede di affrontare un problema di **ricostruzione di
immagini** con un approccio basato su **reti neurali** (deep learning), in contrapposizione
all'approccio *model-based* (Tikhonov, TV, TpV) del Modulo I.

Il setup concreto:
- **Dataset:** immagini mediche **Mayo** (TAC), grayscale, ridimensionate a 256×256.
- **Degradazione:** un operatore di **motion blur** (sfocatura da movimento) noto e fisso,
  più rumore gaussiano.
- **Vincolo:** è *vietato* usare modelli generativi/diffusion o librerie black-box. Si devono
  addestrare reti neurali "da zero" a ricostruire l'immagine pulita dalla misura corrotta.

Le quattro parti richieste:
1. **Data pipeline:** un `Dataset` PyTorch per le immagini Mayo, i dataloader, la
   definizione dell'operatore $K$ (motion blur) e la generazione delle misure corrotte.
2. **Reti:** implementare almeno due architetture — una **SimpleCNN** (predice direttamente
   la ricostruzione) e una **ResCNN** (predice una correzione residuale). La UNet è
   un'estensione opzionale.
3. **Training:** addestrare con `MSELoss`, generando la corruzione *online*, salvare i pesi.
4. **Confronto:** valutare sulle stesse immagini di test con MSE (più PSNR/SSIM), con figure
   comparative, e rispondere a 6 domande di discussione.

---

# 2. La teoria che ci sta dietro

## 2.1 Il problema, in una riga

Come nel Modulo I, il modello di degradazione è:

$$y^\delta = K x + e$$

dove $x$ è l'immagine vera, $K$ l'operatore (qui il motion blur), $e$ il rumore, $y^\delta$
la misura corrotta. L'obiettivo è recuperare $x$ da $y^\delta$.

## 2.2 Il cambio di paradigma: da model-based a data-driven

Nel Modulo I risolvevi, **per ogni immagine**, un problema di minimizzazione
$\min_x \|Kx - y^\delta\|^2 + \lambda \mathcal{R}(x)$.

Nel Modulo II l'idea è opposta: si raccoglie un grande insieme di coppie
$(\text{corrotta}, \text{pulita})$ e si **addestra una rete neurale** $f_\Theta$ a imparare
direttamente la mappa inversa:

$$f_\Theta(y^\delta) \approx x$$

Una volta addestrata, la rete ricostruisce una nuova immagine in un singolo passaggio
(velocissimo), senza ottimizzazione. Il "prior" non è più scritto a mano (come la TV): è
**appreso implicitamente dai dati** e immagazzinato nei pesi $\Theta$ della rete.

## 2.3 Perché servono le reti (e non un modello lineare)

Un modello lineare $f_\Theta(x) = Wx + b$ è troppo poco espressivo. Impilare più strati
lineari non aiuta, perché la composizione di mappe lineari è ancora lineare. La svolta è
inserire **funzioni di attivazione non lineari** (es. ReLU, $\rho(t)=\max(0,t)$) tra gli
strati: questo dà alla rete la proprietà di *approssimazione universale* (può approssimare
qualsiasi funzione continua).

## 2.4 Perché le convoluzioni (CNN)

Per le immagini conta l'informazione **locale** (gruppi di pixel vicini che formano un
oggetto) e **globale**, non quella puntuale (singolo pixel). Un MLP tratta ogni pixel come
isolato → inadatto. La **convoluzione** invece fa scorrere un piccolo kernel $k\times k$
sull'immagine: ogni pixel di output dipende da un intorno locale dell'input. Con il
**padding** (`padding="same"`) l'immagine mantiene la stessa dimensione. Una **CNN** impila
più convoluzioni alternate con ReLU; i suoi parametri sono i numeri dentro i kernel.

Proprietà cruciale: la **translation equivariance** — un oggetto è trattato allo stesso modo
ovunque si trovi nell'immagine. Questo rende le CNN naturali per le immagini.

## 2.5 Il training end-to-end (il cuore dell'homework)

Avendo le immagini pulite $\{x_i\}$ e l'operatore noto $K$, si generano le misure corrotte
**al volo** durante il training:

$$y_i^\delta = K x_i + e$$

e si minimizza la loss supervisionata:

$$\min_\Theta \frac{1}{N}\sum_i \|f_\Theta(y_i^\delta) - x_i\|_2^2 \quad (\text{MSE})$$

Quindi il dataset contiene solo le immagini pulite: la corruzione è prodotta applicando $K$ e
aggiungendo rumore a ogni batch.

> **L'inverse crime.** Aggiungere rumore durante il training è essenziale. Addestrare su dati
> sintetici perfettamente puliti darebbe performance irrealistiche, perché il problema
> diventerebbe troppo facile e scollegato dalla realtà.

L'ottimizzazione usa **SGD/Adam** (passi sui minibatch) e **backpropagation** (regola della
catena attraverso il grafo computazionale) per calcolare i gradienti.

## 2.6 Il residual learning (perché la ResCNN)

Osservazione chiave: i *pattern dell'immagine* sono specifici del dataset (un polmone è
diverso da un fegato), ma l'**artefatto** — la differenza $|x - y^\delta|$ — ha pattern legati
al **tipo di degradazione** (il motion blur), non al contenuto. Quindi:

> **Imparare il residuo (l'artefatto) è più facile e generalizza meglio che imparare
> l'immagine intera.**

La **ResCNN** sfrutta questo: predice la *correzione* e la somma all'input tramite una **skip
connection** (`return out + x`). Parte già dall'immagine corrotta e deve solo "aggiustarla".
Usa Tanh come attivazione finale, perché il residuo può essere positivo o negativo.

## 2.7 UNet (l'estensione opzionale)

Una CNN semplice ha un **campo recettivo** (numero di pixel di input che influenzano un pixel
di output) piccolo: va bene per artefatti *locali*, ma il motion blur è *globale*. La
**UNet** è una CNN multi-scala encoder-decoder: l'encoder riduce la risoluzione (allargando
in fretta il campo recettivo), il decoder la ripristina, e le **skip connection** tra i due
recuperano i dettagli fini. È l'architettura standard per la ricostruzione di immagini ed è
molto più potente (≈1.9M parametri contro i ~10k delle CNN semplici).

---

# 3. Cosa abbiamo implementato

## 3.1 Data pipeline (Parte 1)

`MayoDataset`: legge i file `.png` dalle sottocartelle, li converte in grayscale, li
trasforma in tensori in $[0,1]$ e li ridimensiona a 256×256.

```python
class MayoDataset(Dataset):
    def __init__(self, data_path, data_shape=256):
        self.fname_list = sorted(glob.glob(f'{data_path}/*/*.png'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((data_shape, data_shape)),
        ])
    def __getitem__(self, idx):
        x = Image.open(self.fname_list[idx]).convert('L')
        return self.transform(x)
```

Operatore di degradazione e dataloader:

```python
K = operators.Blurring(img_shape=(256,256), kernel_type='motion',
                       kernel_size=9, motion_angle=20)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

Dataset effettivo: **3306 immagini di training, 327 di test**.

## 3.2 Le reti (Parte 2)

**SimpleCNN** — tre convoluzioni con ReLU, predice direttamente l'immagine:

```python
class SimpleCNN(nn.Module):
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        return self.conv3(h)          # nessuna attivazione finale
```

**ResCNN** — stessa struttura, ma impara il residuo (skip connection + Tanh):

```python
class ResCNN(nn.Module):
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        return self.tanh(self.conv3(h)) + x   # <-- residuo
```

Entrambe hanno **9857 parametri**, così il confronto è equo (stessa capacità, cambia solo
l'idea architetturale). Come estensione è stata implementata anche una **UNet** (≈1.93M
parametri), architettura encoder-decoder multi-scala con skip connection.

## 3.3 Training (Parte 3)

Loop standard: corruzione generata online, `MSELoss`, ottimizzatore Adam, 20 epoche, pesi
salvati in `weights/`. Punti chiave del loop:

```python
y_batch = K(x_batch)                                   # applica il motion blur
y_batch = y_batch + gaussian_noise(y_batch, noise_level=0.01)  # + rumore (anti inverse-crime)
prediction = model(y_batch)
loss = loss_fn(prediction, x_batch)                    # MSE vs immagine pulita
loss.backward(); optimizer.step()                      # backprop + aggiornamento
```

Addestramento su GPU (Colab T4), `noise_level = 0.01`.

## 3.4 Confronto (Parte 4)

Sulla stessa immagine di test corrotta, ricostruzione con entrambe le reti e calcolo di MSE,
PSNR, SSIM, più visualizzazione affiancata (ground truth / corrotta / SimpleCNN / ResCNN).

---

# 4. I risultati

## 4.1 Tabella delle metriche

| Metodo | MSE | PSNR | SSIM |
|--------|-----|------|------|
| Corrotta (motion blur + rumore) | 0.001447 | 28.40 dB | 0.8423 |
| SimpleCNN | 0.000354 | 34.51 dB | 0.9245 |
| ResCNN | 0.000318 | 34.97 dB | 0.9283 |
| **UNet** (opzionale) | **0.000071** | **41.47 dB** | **0.9740** |

## 4.2 Curve di training

Le loss finali (epoca 20): SimpleCNN = 0.000381, ResCNN = 0.000336. Dal grafico si osserva
che la **ResCNN parte molto più in basso** già alla prima epoca (~0.001 contro ~0.0025 della
SimpleCNN) e **resta costantemente sotto** per tutte le 20 epoche. La UNet, addestrata per 10 epoche, scende molto
più rapidamente e più in basso di entrambe (loss finale 0.000064), grazie alla sua capacità
nettamente superiore.

## 4.3 Interpretazione

**Entrambe le reti funzionano bene.** Rispetto all'immagine corrotta, guadagnano circa +6 dB
di PSNR e portano lo SSIM da 0.84 a oltre 0.92. Il motion blur viene rimosso efficacemente.

**La ResCNN vince su tutte e tre le metriche**, a parità di numero di parametri. Questo
conferma sperimentalmente la teoria del residual learning:
- *Convergenza più rapida:* la skip connection fa partire la rete dall'immagine corrotta, che
  è già una buona approssimazione; la rete deve solo imparare la correzione, non ricostruire
  tutto da zero. Questo spiega perché la sua curva di loss parte molto più in basso.
- *Risultato finale migliore:* imparare l'artefatto (legato al task) invece dell'immagine
  (legata al contenuto) porta a un MSE finale più basso e a metriche migliori.

**Il margine è modesto ma sistematico** (MSE 0.000318 vs 0.000354, PSNR 34.97 vs 34.51). Su
un problema relativamente semplice come questo (motion blur moderato, poco rumore) entrambe
le architetture, pur piccole, riescono bene; il vantaggio del residuo si vede soprattutto
nella *dinamica* dell'addestramento (velocità di convergenza) oltre che nel risultato finale.

**La UNet domina nettamente.** Aggiunta come estensione, la UNet (≈1.93M parametri, contro i
~10k delle CNN) raggiunge **41.47 dB di PSNR e 0.9740 di SSIM** — un salto enorme: circa
**+6.5 dB rispetto alla ResCNN** e oltre +13 dB rispetto all'immagine corrotta. L'MSE scende a
0.000071, quasi 5 volte più basso della ResCNN. Anche la convergenza è impressionante: già
all'epoca 2 la loss (0.000220) è migliore di quella finale delle due CNN. Il motivo è
strutturale: il motion blur è un artefatto **globale** (sparge l'informazione di ogni pixel
sui vicini lungo la direzione del movimento), e il grande **campo recettivo** della UNet —
ottenuto col downsampling multi-scala — riesce a "vedere" abbastanza contesto per invertirlo,
mentre le skip connection encoder-decoder recuperano i dettagli fini. Le CNN semplici, con il
loro campo recettivo limitato, non possono competere su un artefatto di questa natura. Il
prezzo è il costo computazionale: ~200× più parametri e training più lento.

## 4.4 Risposte alle domande di discussione

1. **Quale modello è migliore e perché?** La ResCNN, su tutte le metriche. Il residual
   learning rende l'apprendimento più facile perché l'artefatto del motion blur ha pattern
   legati al task, non al contenuto specifico dell'immagine.
2. **Il residual learning ha aiutato?** Sì, in modo misurabile: convergenza più veloce
   (evidente dal grafico) e MSE/PSNR/SSIM finali migliori.
3. **Effetto del rumore:** generato online a `noise_level=0.01`. È necessario per evitare
   l'inverse crime e ottenere stime di performance realistiche; più rumore renderebbe il
   problema più difficile e abbasserebbe le metriche.
4. **Perché un operatore $K$ noto:** inietta la fisica del problema (lo specifico motion
   blur) nel training, così la rete impara la vera mappa inversa di *quel* problema, non una
   generica trasformazione immagine-immagine.
5. **Perché diffidare della sola qualità visiva:** una ricostruzione può sembrare buona ma
   contenere dettagli "allucinati" o degradarsi su dati fuori distribuzione (rumore o blur
   diversi da quelli di training). Le metriche quantitative e i test su condizioni variate
   sono indispensabili.
6. **Come si confronta la UNet con le CNN semplici?** La UNet batte nettamente entrambe le
   CNN (PSNR 41.47 dB vs 34.97 della ResCNN; SSIM 0.974 vs 0.928). Il motivo è il grande
   campo recettivo, ottenuto dal downsampling multi-scala, che le permette di gestire un
   artefatto *globale* come il motion blur, mentre le skip connection encoder-decoder
   recuperano i dettagli fini. Il prezzo è ~200× più parametri (1.93M vs ~10k) e un training
   più lento: un classico compromesso tra qualità e costo computazionale.

---

# 5. Conclusioni

L'homework dimostra il passaggio dal paradigma model-based a quello data-driven per i problemi
inversi di imaging. Con un dataset di immagini pulite e un operatore di degradazione noto, è
possibile addestrare reti neurali end-to-end che ricostruiscono efficacemente immagini
corrotte da motion blur. Il confronto tra SimpleCNN e ResCNN, a parità di capacità, conferma
che incorporare conoscenza strutturale del problema (l'idea che l'artefatto sia più semplice
da apprendere del contenuto) migliora sia la velocità di addestramento sia la qualità finale.
La UNet porta questo principio all'estremo: combinando un campo recettivo ampio (per gli
artefatti globali) e le skip connection (per i dettagli fini), supera nettamente le CNN
semplici (41.5 dB contro ~35), al costo di una capacità ~200 volte maggiore. Il quadro
complessivo illustra il compromesso centrale del deep learning per l'imaging: architetture più
espressive e meglio adattate alla struttura del problema danno ricostruzioni migliori, ma
richiedono più parametri, più dati e più calcolo.

---

*Relazione per l'Homework 1 del Modulo II di Computational Imaging (Davide Evangelista).*
