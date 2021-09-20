---
title: "Parallelizzazione del Metodo Reticolare di Boltzmann per la Simulazione di Fluidi"
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
soluzioni di queste equazioni impiegando metodi numerici appositi. In linea di principio generale,
dato uno stato del fluido al tempo $t$, viene calcolato lo stato (come approssimazione) al tempo
$t+\Delta t$, dove $\Delta t$ e' il passo.
Nella fluidodinamica computazionale esistono diversi metodi utilizzati per il calcolo delle
soluzioni approssimate. Il metodo alla base del solver che e' stato scelto e' il cosiddetto metodo
reticolare di Boltzmann. La motivazione di tale scelta e' che tale metodo e' particolarmente adatto
a sfruttare architetture multicore massive, senza cambiare radicalmente l'implementazione
sequenziale. 

Vestibulum eget neque ac magna sagittis rhoncus. Donec in tincidunt tortor. Duis at mauris aliquet,
dignissim nisi id, tincidunt augue. Nunc varius dui ac luctus varius. Praesent et magna egestas
justo condimentum eleifend sed nec arcu. Quisque vitae quam tincidunt, finibus dui sed, dictum
tellus. Vestibulum libero urna, bibendum nec rhoncus vitae, porta at erat. Donec tempus magna nulla,
eget sollicitudin enim porttitor sed. Fusce sit amet dolor vel nunc sagittis egestas. Nunc at ipsum
id lectus posuere auctor. Mauris lobortis arcu sit amet nulla consequat, elementum fringilla dolor
dignissim.

Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Nulla
vel nulla nisl. Cras vitae consequat tellus, ac posuere metus. Vestibulum consectetur eros erat, vel
posuere nibh sodales a. Donec augue elit, euismod in augue ac, blandit rhoncus odio. Nullam id
suscipit orci. Suspendisse potenti. Proin luctus ipsum in porttitor volutpat. Aliquam erat volutpat.
Proin auctor rhoncus nulla vitae tincidunt. Curabitur feugiat nec arcu vel sollicitudin.

# Sezione 2
Aenean viverra tincidunt molestie. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices
posuere cubilia curae; Cras condimentum ipsum lobortis arcu ultricies porta. Mauris elementum, ante
sit amet dapibus rutrum, sem erat eleifend orci, a vehicula arcu nunc hendrerit diam. Praesent
elementum facilisis augue id fermentum. Proin eleifend nec dolor at dignissim. Suspendisse accumsan
consequat mauris ac auctor. Pellentesque habitant morbi tristique senectus et netus et malesuada
fames ac turpis egestas. Quisque tincidunt nibh in erat lobortis, at dapibus urna imperdiet. Fusce
vitae diam ac nisi auctor egestas sed vitae lectus. Ut consequat massa vel arcu ullamcorper
pellentesque. Nulla rhoncus facilisis purus, a lobortis urna volutpat vitae.

\begin{algorithm}
	\caption{LBM - Compute time step} 
	\begin{algorithmic}
		\For {$x=0$ to $width$}
		  \For {$y=0$ to $height$}
				\State Compute collision step on $l$
        \State Compute streaming step $l \rightarrow l_t$
        \State Swap $l$ with $l_t$
			\EndFor
		\EndFor
	\end{algorithmic} 
\end{algorithm}

Aenean ullamcorper massa in nisi mollis, nec vehicula nibh faucibus. Suspendisse sed interdum nisl.
Pellentesque suscipit metus at congue vestibulum. Cras suscipit nibh non dignissim elementum.
Maecenas lorem nisl, sodales id viverra eu, elementum at lectus. Mauris gravida augue maximus justo
auctor mattis. Nulla vel lorem nisl. Nullam varius commodo leo blandit viverra. Quisque ac malesuada
sem. Duis sem lacus, molestie sed eleifend dapibus, congue sit amet velit. Donec vel metus quis leo
rutrum egestas quis ac elit. Nulla accumsan mauris nec augue venenatis scelerisque. Mauris auctor
aliquet est, at aliquam enim facilisis in. Integer venenatis odio sit amet tortor iaculis rutrum.
Aliquam molestie est fringilla nulla viverra tempus. Pellentesque habitant morbi tristique senectus
et netus et malesuada fames ac turpis egestas.

Aenean vitae nisl sollicitudin, ultrices dui scelerisque, interdum libero. Nulla facilisi. Phasellus
quam dolor, facilisis non ultricies id, vehicula non ligula. Aliquam erat volutpat. Sed vestibulum
odio nunc, vitae iaculis risus semper sed. Mauris vel nisi quis orci tincidunt cursus. Duis
imperdiet sed turpis a consectetur. Pellentesque pharetra pellentesque elementum. Sed vitae sagittis
eros, ut accumsan elit. Duis viverra varius neque nec maximus. Quisque convallis tortor vel pharetra
dignissim. Phasellus ornare et sem vitae porttitor. In imperdiet semper facilisis. Nulla in enim in
tortor ornare euismod a vel nunc.

Proin ut volutpat dolor, iaculis ultrices erat. Curabitur vel lorem facilisis, cursus tellus at,
mollis nisi. Integer varius augue neque. Nunc sodales mattis lacinia. Praesent luctus sapien enim,
in tristique libero bibendum non. Nullam vehicula dolor non vehicula rhoncus. Proin blandit tortor
sit amet urna semper mattis. Cras volutpat suscipit cursus. Etiam sodales leo in justo porta
tristique. Nulla venenatis massa lacus, in tincidunt velit dapibus vel. Suspendisse pharetra arcu et
felis aliquet aliquet ac a dolor. Quisque vitae vehicula felis. Aliquam tincidunt blandit rhoncus.

# Sezione 3
Donec in efficitur enim. Suspendisse potenti. Nam a felis felis. Ut erat sapien, bibendum a
dignissim eget, dapibus tincidunt sem. Aenean dapibus dui non commodo fermentum. Donec lobortis,
lacus ut condimentum luctus, odio arcu auctor magna, quis faucibus ante elit id magna. Vestibulum
leo lectus, vehicula eu mi ut, consectetur aliquam nisl. Nam dapibus congue eros, at vestibulum
tortor. In id ipsum a felis egestas faucibus. Nunc eget quam ac urna elementum ullamcorper id ut
mauris. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos.
Vestibulum id nunc convallis, ultricies purus non, elementum eros.

Quisque eu vulputate lorem. Integer porttitor erat nec urna cursus tincidunt. Vivamus eget nisi et
est tristique feugiat consectetur quis magna. Vestibulum pretium urna et odio pretium iaculis. Nam
dictum, elit ac hendrerit rutrum, ex velit vestibulum est, at egestas mauris urna a tortor. Nullam
sagittis, lacus quis convallis dictum, felis elit vehicula magna, a elementum libero ipsum vel enim.
Nullam sit amet sagittis metus. Cras malesuada a quam a pulvinar. Praesent dolor augue, condimentum
a placerat eget, pellentesque vitae turpis. Mauris mauris lorem, commodo non tellus vel, venenatis
interdum eros. Quisque mattis ipsum id consectetur gravida. Aenean condimentum mollis lorem aliquet
porttitor. Phasellus at dignissim dolor.

Duis non lacus tincidunt, volutpat libero eget, varius velit. Nulla efficitur sagittis sem sit amet
sodales. Cras imperdiet lectus eu ante aliquam fringilla. Phasellus facilisis sollicitudin
venenatis. Mauris fringilla libero sit amet ex iaculis, et ornare sem pulvinar. Lorem ipsum dolor
sit amet, consectetur adipiscing elit. Class aptent taciti sociosqu ad litora torquent per conubia
nostra, per inceptos himenaeos. Sed eu aliquet nisi, vel tempor risus. Mauris gravida eros commodo
nunc ullamcorper, ut congue enim eleifend. Morbi posuere turpis et rhoncus placerat. Nunc orci est,
pellentesque et urna et, eleifend pharetra libero. Integer sollicitudin leo sit amet odio
consectetur, non vestibulum velit laoreet. Nam rhoncus diam erat, vel tristique sem pretium vel.
Quisque pulvinar tincidunt risus, in scelerisque ex cursus vel.

Pellentesque id nunc dapibus, condimentum mi eget, faucibus tortor. Integer eget hendrerit arcu, non
imperdiet enim. Pellentesque eu elit tellus. Donec placerat pretium leo, vitae efficitur tellus
faucibus at. Praesent faucibus ex ac suscipit consequat. Quisque a enim a erat aliquam faucibus non
sed purus. Integer interdum ac dui non posuere. Vivamus pulvinar scelerisque neque, non venenatis
orci pulvinar a. Nulla semper, ante quis feugiat egestas, nunc enim lobortis nulla, eleifend
interdum tortor quam at dolor. Vivamus vestibulum molestie tortor.

In convallis enim in risus dignissim condimentum. Integer arcu enim, sagittis sed ornare eu,
vehicula vitae sapien. Curabitur gravida laoreet sapien, nec porta turpis gravida nec. Nullam non
finibus nulla, vel placerat elit. Nulla in fringilla lorem, in eleifend nisi. Aenean eget ex luctus,
malesuada nulla vel, dictum elit. Duis hendrerit sapien felis, id venenatis ante viverra nec. Nam
bibendum nisl eu hendrerit viverra. Aliquam justo nunc, varius eu nisi in, consequat dapibus enim.
Sed tincidunt, dui sit amet iaculis tempor, mi quam condimentum eros, non laoreet nisl augue
vulputate sapien. Cras venenatis lobortis ex, sit amet condimentum neque iaculis sit amet. Nunc
mattis urna et ultrices placerat. Vivamus justo sapien, commodo vel tortor vel, ullamcorper posuere
nibh.

# Conclusioni
Suspendisse semper ante at ante dictum, ut rutrum mauris pharetra. Integer mi turpis, ultricies a
lacinia sed, fermentum id lacus. Aliquam sed tempor nibh, at tristique mauris. Fusce eget rutrum
eros, vulputate dictum diam. Sed accumsan mollis ligula. Curabitur auctor ligula non egestas
vehicula. Quisque lobortis in felis a tincidunt. Sed a nisi nec massa faucibus iaculis. Nunc diam
velit, aliquam vel mi vel, ornare convallis risus. Sed blandit ipsum eros, et accumsan mi malesuada
quis.

Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Praesent
convallis ante a libero elementum fringilla. Etiam fermentum metus est, in aliquam libero aliquet
suscipit. Donec at porta ante. Nullam sit amet mauris diam. Sed luctus sem nunc, nec pulvinar nibh
fringilla in. Donec enim ipsum, eleifend nec dictum sit amet, scelerisque eu odio.

Integer venenatis enim ac sem lobortis pellentesque. Lorem ipsum dolor sit amet, consectetur
adipiscing elit. Sed non ultricies eros. In hac habitasse platea dictumst. Etiam vitae felis
interdum, egestas magna id, sollicitudin dui. Sed efficitur nulla et arcu eleifend, at imperdiet
quam maximus. Praesent dui diam, elementum a dui vitae, hendrerit vehicula dolor. Mauris sit amet
purus et turpis posuere vehicula vitae eu libero. Etiam sit amet felis finibus, volutpat lacus eu,
volutpat sem. Integer mollis, libero in ultrices finibus, nisl elit feugiat enim, vel laoreet metus
augue sit amet risus. Ut sit amet lacus mi. Phasellus justo metus, porttitor pretium semper
venenatis, vulputate tincidunt enim. Proin aliquam ultricies libero quis fermentum. Donec sed
accumsan nibh. Etiam commodo odio et fringilla eleifend. In dapibus sem nec mauris rutrum, quis
ullamcorper nulla faucibus.
