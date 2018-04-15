# Probabilités - Statistiques : Rapport

Groupe :

- BIARD David
- PENUCHOT Jules
- PICAN Gaëtan

# Exercice 1

Dans l'exercice 3, le classifieur traite chaque pixel un par un. Tous les pixels sont donc considérés comme indépendants. Cela implique que si l'hypothèse est vérifiée, les valeurs de la matrice de covariance devraient etre nulles exepté au niveau de la diagonale où les valeurs devraient être les variances de chaque pixel.

Une manière simple de vérifier cette hypothèse est donc de vérifier toutes les valeurs de la matrice de covariance : si une valeur est non nulle et n'appartient pas à la diagonale, c'est que tous les pixels ne sont pas indépendants.

# Exercice 2

Ici on repart du classifieur Bayésien Naïf Gaussien appliqué aux images MNIST. Ce classifieur implémenté dans l'exercice 3 considère la réalisation d'un pixel conditionnellement à une classe comme une variable aléatoire gaussienne.

Une première analyse que l'on peut effectuer est de regarder, à partir d'un pixel possédant une forte variance, la répartition des valeurs de ce pixel sur toutes nos images. Si l'on représente la répartition de ces valeurs à l'aide d'un histogramme, on peut facilement se rendre compte qu'une courbe représentative telle qu'une gaussienne n'est en fait pas vraiment appropriée pour modéliser cette répartition. En effet, l'hypothèse d'assimiler cela à une seule gaussienne a tendance à fausser la représentation que nous nous étions faite des données.

Une meilleur approche serait de "combiner ou mélanger" deux gaussiennes. Il faudrait donc considérer que la valeur d'un pixel conditiellement à sa classe soit une variable aléatoire dont la distribution est donnée par le mélange de deux gaussiennes.

# Exercice 3

Dans cette exercice, le but est d'obtenir un classifieur naïf bayésien gaussien capable de prédire la classe des images fournies en entré. Dans un premier temps on implémente une version basée sur des images dont les composantes sont représentées par des valeurs réelles. Pour la phase de prédiction on utilise la formule de bayèse afin d'obtenir la probabilité pour chaque image d'appartenir aux 10 classes possible. La probabilité maximum étant choisie. Avec cette méthode on obtient donc un taux de précision d'environ 77 %.

Pour la seconde approche on binarise les images afin que celles-ci soient représentées par des composantes binaires. Les résultats obtenus grâce à cette méthode sont relativement meilleurs avec un taux de précision d'environ 79 %.

Enfin, il est possible d'effecteur une approche légèrement différentes avec un modèle bayesien naïf binomial. Les formules d'estimation et d'inférences sont légèrement différentes, de plus il est possible d'introduire la notion de lissage ce qui permet de ne pas attribuer une valeur nulle sur les pixels qui n'apparaissent jamais. On peut remarquer qu'en "jouant" avec la constante de lissage "alpha" les résultats différent légèrement. En implémentant la méthodes avec l'outils "sklearn" on obtient au mieux un taux d'erreurs d'environ 16 % pour un alpha égal à 0.01. Plus l'on augmente cette constante, la précision du modèle diminue.
