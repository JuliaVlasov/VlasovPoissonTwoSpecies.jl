```@meta
CurrentModule = VlasovPoissonTwoSpecies
```

# VlasovPoissonTwoSpecies


## Description du projet CEMRACS

**Modèles cinétiques pour les plasmas de bord**

L'interaction d’un plasma avec un bord matériel est un problème
très étudié en physique. Lorsqu’un plasma interagit avec une paroi,
une couche mince (appelée ”gaine de Debye”) se forme près du bord
due au déplacement rapide des électrons (plus léger que les ions)
qui sont absorbés par la paroi, créant un déséquilibre de charge
près du bord et ainsi la formation d’une couche limite pour le
potentiel électrique solution de l’équation de Poisson. La modélisation
de ces phénomènes requiert la prise en compte d’équations dites
cinétiques pour chaque espèce (électrons et ions). Dans ce cadre,
une théorie des états d’équilibre a été établie récemment dans le
cas collisionnel [1] et une exploration numérique du cas dynamique
a été effectuée dans [2]. 

L’objectif de ce projet est d’étendre ces
travaux récents en prenant en compte les effets collisionnels ou
de type source volumique au modèle double espèce (ions et électrons)
de Vlasov-Poisson, ce qui s’avère nécessaire pour modéliser les cas
pertinents d’un point de vue de la physique.  Le cœur du projet
sera de construire et d’analyser des méthodes numériques adaptées
pour étudier la dynamique de la formation de gaine en prenant en
compte les effets collisionnels.  Après avoir construit les états
stationnaires, l’objectif est de construire des méthodes numériques
adaptées pour étudier la dynamique des gaines autour de ces états
stationnaires. Plusieurs défis devront être relevés: 
- capturer les effets multi-échelles présents dans le système,
- construire proprement les conditions aux limites pour le potentiel électrique solution de l’équation de Poisson, 
- définir proprement les schémas numériques d’ordre élevé au bord,
- étudier la stabilité des états stationnaires. 

Ainsi, des schémas numériques
d’ordre élevé préservant l’asymptotique quasi-neutre seront développés
dans ce contexte pour pouvoir utiliser des paramètres numériques
indépendants des paramètres physiques (comme la longueur de Debye
qui correspond à la taille de la couche limite ou le rapport massique
entre la masse des électrons et celle des ions) et ainsi réduire
le coût des simulations. Ce travail sera aussi l’occasion de faire
le point sur les techniques mises en place dans la littérature
(physique essentiellement) et de se positionner par rapport à ces
travaux.

Références 

- [1] M. Badsi, M. Campos-Pinto, B. Desprès, A minimization formulation of a bi-kinetic
sheath, 2016, Kinetic Related Models.  
- [2] M. Badsi, M. Merhenberger, L. Navoret, Numerical stability of plasma sheath, 2018, ESAIM Proc.

Le projet est encadré par 
- Mehdi BADSI, Nantes Université, 
- Anaïs CRESTETTO, Nantes Université, 
— Nicolas CROUSEILLES, Inria Rennes - Bretagne Atlantique 
— Michel MEHRENBERGER, Aix-Marseille Université.
