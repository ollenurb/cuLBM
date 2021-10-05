---
title: "Parallelizzazione del Metodo Reticolare di Boltzmann per la Simulazione di Fluidi in due dimensioni"
author:
- Matteo Brunello
- Matr. 858867
reference-section-title: "Bibliografia"
documentclass: article
classoption: twocolumn
papersize: A4
lang: it-IT
numbersections: true
hyperrefoptions:
    - linktoc=all
bibliography: Bibliography.bib
citeproc: true

csl: /home/matteo/.pandoc/csls/ieee.csl
fontsize: 11pt
geometry: "left=2cm,right=2cm,top=2cm,bottom=2cm"
header-includes: |
    \usepackage{algorithm}
    \usepackage{algpseudocode}
    \setlength{\parskip}{0.5em}
    \setlength{\columnsep}{18pt}
    \makeatletter
    \def\@maketitle{%
      \newpage
      \null
      \begin{center}%
      \let \footnote \thanks
        {\LARGE \textbf{\@title} \par}%
        \vskip 1em%
        {\large Relazione di Laboratorio - Sistemi di Calcolo Paralleli e Distribuiti \par}%
        \vskip 1em%
        {\large
          \lineskip .5em%
          \begin{tabular}[t]{c}%
            \@author
          \end{tabular}\par}%
        \vskip 1em%
        {\large \@date}%
      \end{center}%
      \par
      \vskip 1.5em}
    \makeatother
---

# Introduzione
La fluidodinamica computazionale (CFD) e' un campo della meccanica dei fluidi che impiega metodi
numerici e computazionali, per analizzare e risolvere problemi che riguardano il flusso di fluidi.
In questa relazione di laboratorio si discutera' dello sviluppo di un simulatore (o *solver*) di
fluidi in due dimensioni. In particolare, un solver e' un software che utilizza le tecniche della
fluidodinamica computazionale, per simulare il flusso di fluidi (liquidi o gas) e la loro interazione
con le superfici. Verranno discusse due implementazioni del solver: un sequenziale (per cui non
vengono sfruttate tecniche e supporti hardware alla parallelizzazione) e una parallela su GPU
(implementata utilizzando CUDA).
Nella sezione 1 verra' discusso brevemente il metodo che verra' utilizzato per simulare i fluidi,
mentre nella sezione 2 verranno discussi i dettagli implementativi (strutture dati e algoritmi)
della versione sequenziale del solver. Nella sezione 3 si esaminera' l'implementazione parallela
sulla piattaforma CUDA, per cui verranno proposti anche alcuni miglioramenti possibili per eventuali
sviluppi futuri. Infine, nella sezione 4 verranno messe a confronto le due versioni esaminando i
diversi benchmarks eseguiti.

# Il Metodo Reticolare di Boltzmann
Quando di parla di fluidodinamica computazionale e' impossibile non parlare anche delle equazioni di
Navier Stokes. Esse sono un insieme di equazioni parziali differenziali che descrivono il moto dei
fluidi nel tempo. Tipicamente il compito dei solver e' quello di ottenere un'approssimazione delle
soluzioni di queste equazioni per mezzo di metodi numerici appositi.
In linea di principio generale, il compito di un solver e' il seguente: dato lo stato di un fluido
descritto in termini della sua densita' macroscopica $\rho$ e della sua velocita' macroscopica
$\vec{u}$, calcola lo stato risultante al tempo $t+\Delta t$, dove $\Delta t$ e' il passo.
Nella fluidodinamica computazionale esistono diversi metodi e tecniche che possono essere impiegate
per ottenere un'approssimazione di queste soluzioni, ma proprio per la natura delle equazioni
parziali differenziali, molti di questi risultano molto dispendiosi in termini di risorse
computazionali. Il metodo alla base del solver che e' stato scelto, invece, rappresenta un'approccio
alternativo, basato sugli automi cellulari anziche' sulla soluzione delle equazioni di Navier
Stokes. La motivazione principale della scelta e' che per natura risulta particolarmente indicato
per sfruttare architetture multicore massive senza la necessita' di dover modificare radicalmente
l'implementazione sequenziale.
Come il nome suggerisce, questo metodo fu derivato originariamente dalla teoria cinetica dei gas di
Ludwig Boltzmann, secondo la quale i fluidi/gas possono essere immaginati come un grande numero di
particelle che si muovono secondo moti apparentemente casuali. L'idea fondamentale del metodo
reticolare di Boltzmann e' quella di discretizzare lo spazio nel quale queste particelle si muovono,
confinandole ai nodi di un *reticolo*.
In generale, in uno spazio a due dimensioni, le particelle all'interno di un nodo sono limitate a
muoversi in 9 possibili direzioni (inclusa la possibilita' di rimanere stazionarie). Questo modello
descritto a 2 dimensioni e a 9 direzioni possibili e' anche chiamato comunemente modello
`D2Q9`\footnote{Naturalmente esistono altri modelli, quali il D3Q19}.

![Nodo del reticolo del modello D2Q9\label{imgNode}](img/d2q9_node.png){ width=20% }

Le possibili direzioni vengono rappresentate matematicamente mediante 9 vettori $\vec{e_i},
i=0,\dots,8$ a due componenti ($x$, $y$), definiti come:
$$
\vec{e_i} =
\begin{cases}
    (0, 0)&i = 0 \\
    (1, 0), (0, 1), (-1, 0), (0, -1)&i = 1,..4\\
    (1, 1), (-1, 1), (-1, -1), (1, -1)&i = 5,.. 8\\
\end{cases}
$$
Il fluido viene modellato poi mediante una funzione di densita' di probabilita' $f(\vec{x},
\vec{e_i}, t)$, che indica la densita' di fluido alla posizione $\vec{x}$, con direzione
$\vec{e_i}$, al tempo $t$ (tali densita' sono indicate anche come *densita' microscopiche*).
Come detto precedentemente, quello che si vuole ottenere e' lo stato del fluido al tempo $t+\Delta
t$, cioe' dato il valore di $f(\vec{x}, \vec{e_i}, t)$, trovare il valore di $f(\vec{x}, \vec{e_i},
t+\Delta t)$.
Nel metodo reticolare di Boltzmann il calcolo del nuovo stato e' eseguito per mezzo di tre passi
intermedi:

1. Propagazione: le particelle di fluido vengono propagate a seconda della loro direzione ai nodi
   adiacenti
2. Collisione: le particelle di fluido collidono tra loro, di fatto rimodulando la densita'
   all'interno dei singoli nodi
3. Rimbalzo: le particelle di fluido rimbalzano alla collisione con eventuali superfici solide

## Propagazione
Il passo di propagazione consiste essenzialmente nel trasferire la densita' di fluido presente alla
direzione $\vec{e_i}$, di un nodo del reticolo alla posizione $\vec{x}$, alla direzione $\vec{e_i}$
al nodo adiacente corrispondente alla posizione $\vec{x} + \vec{e_i}$. Dal punto di vista del
metodo, il passo e' riassumibile con la seguente equazione:
$$
f(\vec{x} + \vec{e_i}, \vec{e_i}, t + \Delta t) = f(\vec{x}, \vec{e_i}, t)
$$

Il passo e' illustrato anche in Figura \ref{figStreaming}, in cui le frecce piu' spesse indicano il
fluido presente inizialmente nei siti del nodo centrale e la freccia vuota indica il passo di
propagazione.

![Illustrazione del passo di propagazione\label{figStreaming}](img/streaming.png)

## Collisione
Il passo di collisione e' leggermente piu' complicato rispetto al passo di propagazione, poiche'
richiede due passi intermedi in cui vengono calcolate le quantita' macroscopiche $\rho$ e $\vec{u}$.
Dato un istante di tempo $t$ e una cella del reticolo alla posizione $\vec{x}$, e' possibile
calcolare le quantita' macroscopiche citate in precedenza mediante le equazioni seguenti
$$
\begin{aligned}
\rho(\vec{x}, t) &= \sum^{8}_{i=0} f(\vec{x}, \vec{e_i}, t) \\
\vec{u}(\vec{x}, t) &= \frac{1}{\rho} \sum^{8}_{i=0} f(\vec{x}, \vec{e_i}, t) \cdot \vec{e_i}
\end{aligned}
$$
Una volta ottenute le quantita' macroscopiche, e' necessario calcolare il valore della distribuzione
di densita' di equilibrio, data dall'equazione di seguito
$$
f^{eq}(\vec{x}, \vec{e_i}, t) = \rho w_i
[1 + 3\vec{e_i}\cdot \vec{u} + \frac{9}{2}(\vec{e_i} \cdot \vec{u})^2 - \frac{3}{2} |\vec{u}|^2]
$$
In cui $w_i$ indica un peso associato alla direzione *i-esima*. Il peso servea a modellare il fatto
che alcune direzioni siano piu' probabili rispetto ad altre
$$
w_0 = \frac{4}{9}, \quad w_{1, \dots, 4} = \frac{1}{9} \quad w_{5, \dots, 8} = \frac{1}{36}
$$
Una volta ottenuto il valore di $f^{eq}(\vec{x}, \vec{e_i}, t)$ e' possibile infine calcolare per
ogni posizione $\vec{x}$ e direzione $i$ il nuovo stato:
$$
f(\vec{x}, \vec{e_i}, t+ \Delta t) = f(\vec{x}, \vec{e_i}, t) +
\omega [f^{eq}(\vec{x}, \vec{e_i}, t) - f(\vec{x}, \vec{e_i}, t)]
$$
Il termine $\omega$ e' un valore costante, determinato dalla *viscosita'* del fluido (una proprieta'
appartenente ad ogni fluido)

### Rimbalzo
Quando le particelle di fluido collidono con una superficie solida non la devono attraverare, per
cui e' necessario gestire questa condizione in un passo apposito. Tra i diversi metodi esistenti per
gestire tale condizione e' stato scelto il metodo di *bounce-back*.

![Passo di rimbalzo\label{figBounceBack}](img/bb_ultimate.png){ width=40% }

In sintesi, se in una cella $\vec{x}$ del reticolo e' presente un ostacolo - che viene indicato da
una variabile booleana - allora cio' che accade e' che il fluido viene "rimbalzato" verso la
direzione opposta.

# Implementazione Sequenziale
In questa sezione verra' discussa l'implementazione sequenziale in cui verranno anche illustrate
ottimizzazioni a livello di compilazione (`O3`)

L'implementazione attuale del simulatore non richiede particolari trattazioni
dal punto di vista algoritmico.
Per eseguire uno step di simulazione, l'algoritmo esegue sequenzialmente i passi
descritti in precedenza. I passi stream e collide in particolare devono essere
eseguiti in sequenza, poiche' il passo di propagazione eseguito su una cella
arbitraria, necessita di modificare i valori delle celle adiacenti.
$$
collide \rightarrow stream \rightarrow bounce
$$
Nell'implementazione, il modello `D2Q9` e' stato rappresentato utilizzando una
matrice di dimensioni $larghezza \times altezza$ di nodi di reticolo. Ogni nodo
contiene il valore di $f$ in tutte le 9 direzioni e il valore del vettore a due
componenti $\vec{u}$.
```c
typedef Real float;
#define Q 9

typedef struct LatticeNode {
    Real[Q] f;
    Vector2D<Real> u = {0, 0};
} LatticeNode;

LatticeNode[WIDTH][HEIGHT] lattice;
```
Siccome i 3 steps necessitano dei valori di $f$ al passo precedente, vengono
mantenute in memoria due reticoli: uno ($l$) che conterra' i valori al tempo
$t$, mentre l'altro ($l'$) al tempo $t + \Delta t$. Alla fine di ogni time-step,
i puntatori dei relativi reticoli vengono scambiati.  Per poter visualizzare la
simulazione, lo stato viene poi scritto all'interno di un file VTK in modo da
poter essere usufruito dal software di visualizzazione
Paraview\textsuperscript{\textcopyright}.

\begin{algorithm}
    \caption{Passo di simulazione}
    \begin{algorithmic}
        \State Compute $Collide$ step on $l$
        \State Compute $Streaming$ step $l \rightarrow l'$
        \State Compute $Bounce$ step on $l'$
        \State Swap $l$ with $l'$
    \end{algorithmic}
\end{algorithm}

Nello specifico, il passo di collisione dovra' calcolare la nuova densita' per
ogni cella del reticolo per ogni direzione. Il passo e' riassunto per mezzo
dell'algoritmo seguente.

\begin{algorithm}
    \caption{Passo di Collisione}
    \begin{algorithmic}
        \For {$x=0$ to $width$}
            \For {$y=0$ to $height$}
                \State $\vec{x}_{pos} = (x, y)$
                \State Compute $\rho(\vec{x}_{pos})$ and $\vec{u}(\vec{x}_{pos})$
                \For {i = 0 to Q}
                    \State Compute $f^{eq}(\vec{x}_{pos})$ using $\rho$ and
                    $\vec{u}$
                    \State $f_i(\vec{x}_{pos}) \mathrel{+}= \Omega
                    [f^{eq}_i(\vec{x}_{pos}) - f_i(\vec{x}_{pos})]$
                \EndFor
            \EndFor
        \EndFor
    \end{algorithmic}
\end{algorithm}

I cicli piu' esterni servono a scorrere l'intero spazio bidimensionale, in cui
in ogni cella vengono calcolate le quantita' macroscopiche necessarie per il
calcolo di $f^{eq}$. Nel ciclo piu' interno vengono infine calcolate le nuove
densita' associate ad ogni direzione $i$.

\begin{algorithm}
    \caption{Passo di Propagazione}
    \begin{algorithmic}
        \For {$x=0$ to $width$}
            \For {$y=0$ to $height$}
                \For {i = 0 to Q}
                    \State $\vec{x}_{pos} = (x, y)$
                    \State $f'_i(\vec{x}_{pos} + \vec{e}_i) = f_i(\vec{x}_{pos})$
                \EndFor
            \EndFor
        \EndFor
    \end{algorithmic}
\end{algorithm}

Il passo di propagazione e' invece piu' semplice, e consiste essenzialmente
nello scorrere l'intero spazio del reticolo e aggiornare opportunamente le nuove
densita' associate ad ogni direzione nel reticolo $f'$.

# Implementazione Parallela su GPU
La parallelizzazione di simulazioni basate sugli automi cellulari e' un problema
generalmente conosciuto e di facile parallelizzazione dal momento che si
tratta di computazioni sincrone. Come ben noto, tali computazioni sono
caratterizzate da un punto di sincronizzazione per tutti i proccessi coinvolti,
tipicamente inserito prima della fine di un passo di simulazione.
Il modello di programmazione scelto per la parallelizzazione del simulatore e'
*Single Program Multiple Threads* (SPMT), supportato dalla piattaforma di
programmazione CUDA. Sfruttando l'indipendenza dei dati tra celle del reticolo
e' possibile "mappare" idealmente ogni thread ad ogni cella. Da questo punto di
vista, quindi, i threads non necessitano di nessuna sincronizzazione a livello
di threads block, rendendo piu' semplice l'implementazione. L'unica dipendenza
e' evidenziata dalla necessita' dei 3 sotto-passi di simulazione discussi in
precedenza di essere eseguiti in modo sincrono. Per poter sincronizzare tutti i
thread blocks tra di loro, i 3 sotto-passi di simulazione sono stati
implementati come CUDA kernels. Il passo di simulazione, quindi, consistera'
semplicemente in una chiamata ripetuta dei kernels in sequenza e di una
successiva sincronizzazione a livello di device. Alla fine dei sotto-passi, si
trasferiscono i dati del reticolo dal device all'host. Anche in questo caso, i
puntatori dei reticoli $ld$ e $ld$ (allocati sul device) vengono scambiati tra
loro.

\begin{algorithm}
    \caption{Passo di Simulazione Parallelo}
    \begin{algorithmic}
        \State Launch $CollideKernel$ step on $ld$
        \State Synchronize with device
        \State Launch $StreamKernel$ step $ld \rightarrow ld'$
        \State Synchronize with device
        \State Launch $BounceKernel$ step on $ld'$
        \State Transfer memory back to host $ld' \rightarrow l$
        \State Swap $ld$ with $ld'$
    \end{algorithmic}
\end{algorithm}

La parallelizzazione dei singoli passi, invece, consiste nell'eliminazione dei
due cicli for dell'implementazione sequenziale necessari a scorrere l'intero
spazio bidimensionale della simulazione. Grazie al modello di programmazione
scelto, tale parallelizzazione e' molto semplice, per cui basta eliminare i due
cicli `for` e sostituire gli indici con l'indice $x$ e $y$ di thread all'interno
del blocco. Di seguito e' riportato solo il passo di collisione, siccome tutti
gli altri passi sono stati parallelizzati seguendo lo stesso principio

\begin{algorithm}
    \caption{Passo di Collisione Parallelo}
    \begin{algorithmic}
        \State $x = blockIdx.x * blockDim.x + threadIdx.x$
        \State $y = blockIdx.y * blockDim.y + threadIdx.y$
        \State $\vec{x}_{pos} = (x, y)$
        \State Compute $\rho(\vec{x}_{pos})$ and $\vec{u}(\vec{x}_{pos})$
        \For {i = 0 to Q}
            \State Compute $f^{eq}(\vec{x}_{pos})$ using $\rho$ and
            $\vec{u}$
            \State $f_i(\vec{x}_{pos}) \mathrel{+}= \Omega
            [f^{eq}_i(\vec{x}_{pos}) - f_i(\vec{x}_{pos})]$
        \EndFor
    \end{algorithmic}
\end{algorithm}

# Risultati e benchmarks
Per poter eliminare ogni interferenza dovuta a tempi di esecuzione sincroni,
ogni test e' stato effettuato evitando di eseguire il passo di scrittura dello
stato all'interno dei files per la visualizzazione. In questo modo, i tempi di
esecuzione comprendono solo i tempi di computazione e di trasferimento in
memoria (nel caso dell'implementazione parallela, anche dei tempi di
trasferimento dati *device-host*).

# Conclusioni
In questa sezione verranno messe le conclusioni tratte dagli esperimenti
