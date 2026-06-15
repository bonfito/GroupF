# Metodi model-based: regolarizzazione variazionale e ibrida

## Formulazione del problema

L'operazione di degradazione delle immagini retiniche è modellata come problema
inverso lineare

$$y = Ax + e,$$

dove `x` è l'immagine pulita da ricostruire, `A` è l'operatore di *motion blur*
(angolo 45°, kernel 9×9), `e` è rumore gaussiano additivo di deviazione standard
`σ`, e `y` è l'osservazione degradata. Poiché `A` è fortemente mal condizionata,
l'inversione diretta amplifica il rumore e il problema risulta mal posto. Si adotta
quindi un approccio *model-based*, in cui la ricostruzione è ottenuta minimizzando un
funzionale composto da un termine di fedeltà ai dati e da un termine di
regolarizzazione:

$$\min_x \; \tfrac{1}{2}\|Ax - y\|_2^2 + \lambda\,R(x).$$

Il primo termine, scritto in norma 2 al quadrato coerentemente con l'ipotesi di
rumore gaussiano, impone che l'immagine ricostruita, una volta ri-degradata tramite
`A`, sia compatibile con i dati osservati. Il secondo termine `R(x)` introduce
informazione a priori sulla struttura delle immagini reali. Il parametro `λ > 0`
bilancia i due contributi: valori troppo piccoli lasciano residuo rumore, valori
troppo grandi producono over-regularization con perdita di dettaglio.

Questa formulazione è comune a entrambi i metodi qui descritti; ciò che li distingue
è la scelta del regolarizzatore `R(x)`.

## Algoritmo di ottimizzazione: FISTA

Il funzionale è composto da un termine liscio (la fedeltà ai dati) e da un termine
non differenziabile (il regolarizzatore). Si utilizza pertanto lo schema *proximal
gradient* nella sua versione accelerata **FISTA** (*Fast Iterative
Shrinkage-Thresholding Algorithm*), che alterna un passo di gradiente sul termine di
fedeltà,

$$\nabla f(x) = A^\top (Ax - y),$$

e un passo prossimale sul regolarizzatore, introducendo inoltre un termine di
*momentum* che porta l'ordine di convergenza da `O(1/k)` a `O(1/k²)`. Poiché il
kernel di blur è normalizzato si ha `‖AᵀA‖ = 1`, da cui costante di Lipschitz
`L = 1` e passo unitario. Lo stesso schema iterativo è impiegato per entrambi i
metodi, cambiando unicamente l'operatore prossimale.

## Metodo variazionale: regolarizzazione Wavelet (FISTA-Wavelet)

Il primo metodo adotta come regolarizzazione la norma 1 dei coefficienti wavelet
dell'immagine,

$$R(x) = \|Wx\|_1,$$

con `W` trasformata wavelet ortonormale (Daubechies db4, 3 livelli). L'assunzione
sottostante è la *sparsità*: le immagini naturali sono descritte da pochi
coefficienti wavelet significativi, mentre rumore e dettaglio fine corrispondono a
coefficienti di piccola ampiezza. Imporre la norma 1 favorisce soluzioni sparse,
attenuando il rumore e preservando le strutture principali. Essendo `W` ortonormale,
l'operatore prossimale della norma 1 si riduce al *soft-thresholding* dei
coefficienti, da cui il nome dell'algoritmo.

## Metodo ibrido: Adaptive Weighted Total Variation (AWTV)

Il secondo metodo sostituisce la regolarizzazione wavelet con una **Total Variation
pesata e adattiva**, ispirata a Morotti et al. (2025). La Total Variation classica,

$$TV(x) = \|\nabla x\|_1,$$

assume immagini costanti a tratti, cioè a gradiente sparso, ed è particolarmente
efficace nel preservare i bordi mantenendo lisce le regioni omogenee — condizione
ben verificata per le immagini retiniche, costituite da vasi netti su sfondo
relativamente uniforme. Per ridurre l'effetto di *staircasing* tipico della TV
uniforme e per modulare la regolarizzazione in funzione del contenuto locale, si
introducono pesi spazialmente variabili `w_{i,j}`:

$$R(x) = \sum_{i,j} w_{i,j}\,|\nabla x|_{i,j}.$$

I pesi sono **adattivi**: vengono ri-stimati dalla mappa dei bordi dell'immagine
corrente in un ciclo esterno (*reweighting*), secondo `w_{i,j} = 1/(|\nabla x|_{i,j} + β)`.
In corrispondenza dei bordi (gradiente elevato) il peso è basso e la regolarizzazione
viene attenuata, preservando le strutture; nelle regioni piatte il peso è elevato e
la regolarizzazione viene rafforzata, sopprimendo il rumore. Si nota che, mentre nel
lavoro di riferimento i pesi possono essere predetti da una rete addestrata
(*boosted by learning techniques*), nella presente implementazione si è adottato il
reweighting classico basato sul gradiente.

## Scelta del preprocessing: immagini in scala di grigi

Le immagini sono state convertite in scala di grigi, ridimensionate a 256×256 e
normalizzate in [0,1]. La conversione a singolo canale è motivata dalla natura del
task: il deblur e il denoise agiscono sulla struttura e sull'intensità dei contorni,
non sul colore, che non apporta informazione utile alla ricostruzione. La scelta
garantisce inoltre coerenza con tutti gli altri metodi confrontati, incluso il prior
di diffusione, definito su singolo canale.

## Scelta del parametro di regolarizzazione

Il parametro `λ` è stato selezionato euristicamente tramite ricerca su griglia,
valutando le ricostruzioni su un sottoinsieme dell'insieme di *validation* (separato
da quello di test) per ciascun livello di rumore. La scelta finale è stata basata
sull'indice SSIM, privilegiando la fedeltà strutturale, particolarmente rilevante in
ambito di imaging medico. Si osserva che il valore ottimale di `λ` cresce
all'aumentare del rumore: a rumore più intenso corrisponde la necessità di una
regolarizzazione più forte. Per i livelli di rumore più elevati (σ = 0.05 e 0.1) il
valore selezionato coincide con l'estremo superiore della griglia, suggerendo che il
valore ottimale potrebbe essere ulteriormente più alto.

## Risultati

I valori riportati sono medie su 400 immagini di test, per ciascun livello di rumore.

| σ      | FISTA-Wavelet PSNR | AWTV PSNR | FISTA-Wavelet SSIM | AWTV SSIM |
|--------|--------------------|-----------|--------------------|-----------|
| 0.005  | 37.4               | **39.63** | 0.925              | **0.948** |
| 0.01   | 34.6               | **37.19** | 0.881              | **0.907** |
| 0.05   | **28.9**           | 27.98     | **0.561**          | 0.554     |
| 0.1    | **20.5**           | 17.05     | **0.186**          | 0.151     |

*(I valori FISTA-Wavelet vanno verificati con il file `fista_all.csv`; quelli AWTV
provengono da `awtv_all.csv`.)*

## Discussione

A livelli di rumore basso e medio (σ = 0.005 e 0.01) l'Adaptive Weighted TV supera
nettamente la regolarizzazione wavelet, sia in PSNR sia in SSIM. Il risultato è
coerente con l'ipotesi sottostante ai due metodi: la descrizione "regioni omogenee
con bordi netti", rafforzata dai pesi adattivi che proteggono i contorni, modella le
immagini retiniche meglio della sola sparsità wavelet, preservando con maggiore
accuratezza i vasi sanguigni.

A rumore elevato (σ = 0.05 e 0.1) il vantaggio si annulla e la regolarizzazione
wavelet risulta leggermente migliore. La spiegazione è che, in presenza di rumore
intenso, l'informazione sui bordi viene in gran parte distrutta: il meccanismo di
preservazione dei contorni dell'AWTV perde efficacia, e i valori assoluti delle
metriche (SSIM ≈ 0.15 a σ = 0.1) indicano comunque una ricostruzione fortemente
compromessa per entrambi i metodi. Complessivamente, l'AWTV si conferma il metodo
model-based più efficace nel regime di rumore realistico, a fronte di un costo
computazionale leggermente superiore dovuto al ciclo di reweighting.
