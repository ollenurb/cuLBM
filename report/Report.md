---
title: "Sistemi di Calcolo Paralleli e Distribuiti"
subtitle: "Relazione di Laboratorio"
author:
- Matteo Brunello 
- Matr. 858867
reference-section-title: "Bibliografia"        
documentclass: article 
papersize: A4
lang: it-IT
hyperrefoptions:
    - linktoc=all
bibliography: Bibliography.bib
citeproc: true

csl: /home/matteo/.pandoc/csls/ieee.csl
fontsize: 12pt
geometry: "left=2cm,right=2cm,top=1cm,bottom=2cm"
header-includes: |
    \makeatletter
    \def\@maketitle{%
      \newpage
      \null
      \vskip 2em%
      \begin{center}%
      \let \footnote \thanks
        {\LARGE \@title \par}%
        \vskip 1em%
        {\large \emph{Parallelizzazione del metodo reticolare di Boltzmann per la simulazione di fluidi} \par}%
        \vskip 1.5em%
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
Il progetto di laboratorio finale di cui si discutera' in questa relazione, consiste nello sviluppo
di un simulatore (o *solver*) di fluidi in due dimensioni. Piu' in particolare, verranno discusse
due implementazioni del solver: una sequenziale (per cui non vengono sfruttate tecniche e supporti
hardware alla parallelizzazione) e una parallela su GPU (implementata utilizzando CUDA).

Per poter simulare efficientemente i
fluidi e sfruttare al meglio architetture multicore massive (quali acceleratori grafici), e' stato
deciso di basare il solver sul cosiddetto "*metodo reticolare di Boltzmann*"
