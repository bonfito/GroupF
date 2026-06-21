# Homework 1 — Model-Based Reconstructions
## Riassunto completo: teoria da zero + interpretazione dei risultati

> Documento pensato per essere letto da chi parte da zero e arriva fino alla
> discussione dei risultati ottenuti nei quattro task.

---

# PARTE 1 — LA TEORIA, PARTENDO DA ZERO

## 1.1 L'idea di fondo: il problema inverso

Quando un sensore acquisisce un'immagine (una fotocamera, un microscopio, una TAC),
non registra mai la realtà perfetta. Registra una versione **degradata**: con rumore,
sfocata, rimpicciolita, o sotto forma di proiezioni.

Chiamiamo:
- $x$ = l'immagine vera (la realtà che vogliamo ricostruire, *ground truth*)
- $y$ = il dato che il sensore ci dà davvero (degradato)
- $A$ = l'operatore che descrive *come* il sensore degrada l'immagine
- $e$ = il rumore casuale aggiunto

Il modello matematico, valido per **tutti e quattro i task**, è:

$$y = A x + e$$

Il *Computational Imaging* fa il percorso inverso del sensore: parte da $y$ per
ricostruire $x$. Per questo si chiama **problema inverso**.

**La differenza tra i quattro task è solo cosa rappresenta $A$:**

| Task | Operatore $A$ | Cosa fa |
|------|---------------|---------|
| 1. Denoising | Identità | niente, solo rumore aggiunto |
| 2. Deblurring | Convoluzione (blur) | sfoca l'immagine |
| 3. Super-resolution | Downsampling | rimpicciolisce l'immagine |
| 4. CT | Trasformata di Radon | proietta a raggi X (sinogramma) |

## 1.2 Il problema: perché non basta "invertire"?

Verrebbe da pensare: se $y = Ax + e$, basta calcolare $x = A^{-1}y$. **Non funziona.**

Il problema è **mal posto**: l'inversione diretta *amplifica il rumore* in modo
catastrofico. Anche un disturbo minuscolo in $y$ diventa, dopo l'inversione, un errore
enorme in $x$. Tecnicamente: l'operatore $A$ ha dei "valori singolari" molto piccoli,
e invertire significa *dividere* per quei numeri piccoli, facendo esplodere le
componenti ad alta frequenza (dove vivono sia i bordi che il rumore).

## 1.3 La soluzione: la regolarizzazione

Invece di invertire alla cieca, riformuliamo il problema come una **minimizzazione**:

$$\hat{x} = \arg\min_x \underbrace{\|A x - y\|_2^2}_{\text{(1) fedelta' ai dati}} + \lambda \underbrace{R(x)}_{\text{(2) regolarita'}}$$

- Il **primo termine** dice: "resta coerente con quello che ho misurato".
- Il **secondo termine** $R(x)$ dice: "ma assomiglia a un'immagine naturale plausibile".
- $\lambda$ è la **manopola** che bilancia i due: il peso della regolarizzazione.

Questa informazione in più ($R(x)$) è ciò che stabilizza il problema e lo rende risolvibile.

## 1.4 I tre regolarizzatori

Tutti e tre guardano il **gradiente** dell'immagine (quanto cambia bruscamente il colore
tra pixel vicini): piccolo nelle zone piatte, grande sui bordi e sul rumore.

### Tikhonov (norma L2): $R(x) = \|\nabla x\|_2^2$
Penalizza i cambiamenti **al quadrato**. Smussa tutto in modo uniforme.
- ✅ Stabile, semplice, ottimo controllo del rumore
- ❌ Sfoca i bordi (non distingue un bordo vero dal rumore)
- Problema **convesso**: una sola soluzione, facile da trovare

### Total Variation (TV, norma L1): $R(x) = \|\nabla x\|_1$
Penalizza i cambiamenti in modo **lineare**. Questo rende il gradiente "sparso":
elimina le tante piccole variazioni (rumore) ma tiene i pochi salti grandi (bordi).
- ✅ Mantiene i bordi netti
- ❌ Effetto "cartone animato" (*staircasing*): le sfumature diventano zone piatte
- Problema **convesso** ma non differenziabile → serve Chambolle-Pock

### Total-p Variation (TpV, norma Lp con p=0.5): $R(x) = \|\nabla x\|_p^p$
Versione estrema della TV, per bordi ancora più netti.
- ✅ Potenzialmente il più nitido
- ❌ **Non convesso**: instabile, può incastrarsi in minimi locali
- Il più imprevedibile

## 1.5 La scelta di lambda: due criteri

**A) Best-metric ("col senno di poi").** Provo tanti lambda e scelgo quello col PSNR
piu' alto. Funziona solo in laboratorio perche' richiede di conoscere $x$ (la realta').

**B) Discrepancy Principle ("onesto").** Non guardo la realta'. Scelgo lambda in modo
che l'errore residuo $\|A\hat{x} - y\|$ sia grande quanto il rumore atteso $\delta = \|e\|$.
Idea: non ha senso "pulire" piu' del rumore presente. Non richiede la ground truth.

## 1.6 Le due metriche di valutazione

- **PSNR** (dB): misura l'errore numerico pixel per pixel. Piu' alto = meglio.
- **SSIM** (0-1): misura la somiglianza *strutturale*, piu' vicina alla percezione umana.

Si usano insieme perche' spesso non concordano: un metodo puo' vincere su una e perdere
sull'altra. Questo accade tipicamente tra Tikhonov (vince PSNR) e TV (vince SSIM).

---

# PARTE 2 — I RISULTATI DEI QUATTRO TASK

> I valori sono quelli ottenuti sulle immagini di test reali. La cosa importante non
> sono i numeri assoluti, ma il **pattern** e il **perche'**.

## Task 1 — Denoising

| Metodo | lambda | PSNR | SSIM |
|--------|--------|------|------|
| Noisy | - | 25.97 | 0.6615 |
| Tikhonov (best/DP) | 0.05 | 27.41 | 0.7154 |
| TV p=1 (best) | 0.05 | 28.05 | **0.8344** |
| TpV p=.5 (best) | 0.01 | **28.73** | 0.8059 |

**Lettura:** caso piu' facile (l'operatore non degrada, solo rumore). Tutti migliorano.
La **TpV vince sul PSNR**, ma la **TV ha lo SSIM piu' alto** — segno che la TV ricostruisce
meglio la *struttura* percepita, mentre la TpV ottimizza l'errore numerico introducendo
qualche artefatto. Tikhonov rimuove il rumore ma resta il piu' sfocato.

## Task 2 — Deblurring

| Metodo | lambda | PSNR | SSIM |
|--------|--------|------|------|
| Blurred | - | 25.92 | 0.7768 |
| Tikhonov (best/DP) | 0.001 | **30.27** | **0.8662** |
| TV p=1 (best/DP) | 0.01 | 28.97 | 0.8528 |
| TpV p=.5 (best) | 0.005 | 27.61 | 0.7971 |

**Lettura:** qui **vince Tikhonov**, su entrambe le metriche! Questo e' il risultato
piu' istruttivo: *non esiste un metodo migliore in assoluto*. Nel deblurring il problema
e' dominato dall'amplificazione del rumore alle alte frequenze, e lo smoothing dolce di
Tikhonov lo controlla in modo molto stabile. La TV, con il suo effetto a blocchi, su
un'immagine con sfumature morbide perde leggermente. La regolarizzazione qui e'
**vitale**: la sfocatura cancella informazione, e senza prior il rumore esploderebbe.

## Task 3 — Super-Resolution (fattore x2)

| Metodo | lambda | PSNR | SSIM |
|--------|--------|------|------|
| Bicubic (baseline) | - | **32.04** | 0.9163 |
| Tikhonov (best/DP) | 0.001 | 29.33 | 0.8741 |
| TV p=1 (best) | 0.01 | 30.48 | 0.8972 |
| TpV p=.5 (best) | 0.05 | 30.53 | 0.8946 |

**Lettura:** qui il **baseline bicubico vince sul PSNR**. Non e' un fallimento: con un
fattore di downscaling piccolo (x2) e poco rumore, il problema e' facile e l'interpolazione
classica e' gia' molto efficace. I metodi model-based brillano quando il problema e' piu'
difficile (fattore x4, piu' rumore). **Suggerimento:** rifarlo con `downscale_factor=4` e
`noise_level=0.03` per far emergere il vantaggio di TV/TpV — mostrare due regimi (uno facile
dove vince il bicubico, uno difficile dove vincono i model-based) e' un'analisi che fa colpo.

## Task 4 — CT Reconstruction (60 angoli)

| Metodo | lambda | PSNR | SSIM |
|--------|--------|------|------|
| FBP (baseline) | - | 18.93 | 0.3718 |
| Tikhonov (best/DP) | 0.001 | **26.57** | 0.6767 |
| TV p=1 (best/DP) | 0.5 | 26.09 | 0.6692 |
| TpV p=.5 (best) | 0.5 | 25.30 | 0.6648 |

**Lettura:** qui si vede il **salto piu' grande** grazie alla regolarizzazione. La FBP
(il metodo analitico classico della TAC, non regolarizzato) si ferma a 18.93 dB con SSIM
0.37: immagine riconoscibile ma piena di artefatti a strisce, perche' con soli 60 angoli
il problema e' gravemente sotto-determinato. I metodi model-based salgono tutti a ~26 dB:
la regolarizzazione non "migliora" la ricostruzione, la **rende possibile**. Nota che TV e
TpV scelgono lambda=0.5 (il massimo della griglia): il problema "chiede" molta
regolarizzazione perche' mancano tanti dati. Tikhonov in testa al PSNR, coerente con la
stabilita' dello smoothing L2 su un problema cosi' mal posto.

---

# PARTE 3 — IL QUADRO D'INSIEME (la cosa da dire all'esame)

## Tre messaggi chiave

**1. Stesso impianto, operatore diverso.** Tutti e quattro i task sono lo stesso problema
inverso $y = Ax + e$ risolto con $\min \|Ax-y\|^2 + \lambda R(x)$. Cambia solo $A$
(Identita' / blur / downscaling / Radon) e la difficolta'. Questo e' il filo conduttore.

**2. Non esiste "il metodo migliore" in assoluto.** I risultati lo dimostrano:
- Denoising → TpV sul PSNR, TV sullo SSIM
- Deblurring → Tikhonov
- Super-resolution → bicubico (in regime facile)
- CT → Tikhonov sul PSNR, ma tutti i model-based stracciano la FBP

La scelta dipende dal problema, dall'immagine e dalla metrica che ti interessa.

**3. PSNR vs SSIM raccontano cose diverse.** Tikhonov tende a vincere il PSNR (errore
numerico) perche' smussa; TV tende a vincere lo SSIM (struttura) perche' tiene i bordi.
Mostrare entrambe, e commentare quando divergono, dimostra di aver capito la teoria.

## Il ruolo della difficolta' del problema

C'e' una progressione chiara nei quattro task: piu' il problema e' mal posto, piu' la
regolarizzazione e' indispensabile.
- **Denoising:** regolarizzazione *utile* (tutti migliorano di ~2-3 dB)
- **Deblurring:** regolarizzazione *vitale* (l'inversione diretta esploderebbe)
- **Super-resolution:** regolarizzazione utile, ma in regime facile il baseline basta
- **CT a pochi angoli:** regolarizzazione *salvavita* (+7 dB sulla FBP, da inutilizzabile a buona)

## Sui due criteri per lambda

Nei task con rumore basso, Discrepancy Principle e best-PSNR spesso scelgono lo stesso
lambda (la griglia e' grossolana rispetto al poco rumore). Nei casi con piu' rumore o
griglia piu' fine, divergono: il Discrepancy Principle (che NON usa la ground truth)
tende a scegliere un lambda piu' regolarizzante rispetto all'ottimo "oracolo". Questa
differenza e' il senso stesso della richiesta: nella realta' non hai la ground truth,
quindi un criterio "onesto" come il DP e' cio' che useresti davvero.

---
