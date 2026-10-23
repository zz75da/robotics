# Marso Robotics — Data Lab local

Prototype Windows, local et open source pour explorer des journaux/télémétries de robots, tester un traitement vidéo et exercer un cycle MLOps. Il ne requiert ni compte cloud, ni serveur de base de données externe, ni poids de modèle pré-entraîné ; SQLite, les artefacts du modèle et les rapports Evidently restent locaux. L'interface est publiée exclusivement sur `127.0.0.1`.

## Architecture du prototype

```text
Navigateur (HTML/CSS/JS sans framework)
        │ HTTP sur loopback uniquement
        ▼
Serveur Python standard (app.py)
   ├── CSV / JSONL → KPI en mémoire (robot_data.py)
   ├── Jeu synthétique → ZIP généré à la demande
   └── Vidéo → OpenCV → MP4 téléchargé (video_privacy.py)
        │
        ├── Ordonnanceur à un processus → SQLite (operations.py)
        ├── Logs JSONL avec rotation locale (observability.py)
        ├── MLOps : scikit-learn → modèle versionné → API de prédiction
        ├── Evidently : rapports de dérive sans conservation des données sources
        └── Métriques Prometheus → Prometheus → dashboard Grafana provisionné
```

Les journaux et vidéos importés ne sont pas conservés après la requête. Le jeu de démonstration est synthétique : 600 lignes de télémétrie, incidents JSONL, labels d'anomalie artificiels équilibrés à 5 % et une courte scène dessinée sans personne réelle. Le pipeline planifié est lui aussi une **démonstration synthétique** ; il analyse les événements et réévalue le candidat du modèle, sans surveiller un répertoire métier. SQLite conserve les exécutions (500) et les lots de prédiction (500). Les artefacts du modèle sont conservés à 3 versions promues maximum, les rapports Evidently à 500, les logs tournent sur quatre fichiers de 2 Mio (8 Mio) et Prometheus est retenu à 7 jours et 512 Mio. Docker est facultatif pour lancer l'application seule, mais requis pour Grafana, Prometheus et l'interface Evidently. L'environnement Python installé pour les tests occupe environ 0,8 Gio sur le PC utilisé pour cette validation. L'empreinte des images/cache Docker n'a pas pu être mesurée, car le moteur Docker n'était pas démarré ; pour vérifier le total sur votre PC après le premier démarrage, exécutez `docker system df`.

## Démarrage sur Windows

Python 3.10 à 3.12 64 bits est recommandé.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Ouvrir <http://127.0.0.1:8765>. Pour choisir un autre port, définir `ROBOTICS_DEMO_PORT` avant `python app.py`. Pour télécharger des échantillons, cliquer sur « Générer et télécharger le jeu de test » ; le ZIP produit contient le CSV, le JSONL, la vidéo AVI et un descriptif. Le ZIP est construit à la demande, pas stocké sur disque.

### Télécharger une vidéo par lien direct

Le script autonome `download_video.py` accepte une URL HTTPS directe vers un fichier vidéo (.mp4, .webm, .mov, .avi, .mkv ou .m4v), de 250 Mo maximum. Lancer `python download_video.py` puis coller cette URL ; le fichier est enregistré dans `downloads/`. Il ne télécharge pas les pages YouTube (`youtu.be` inclus) et ne contourne pas les restrictions d'une plateforme. Téléchargez uniquement des fichiers que vous êtes autorisé à obtenir.

### Stack MLOps avec Docker Compose

Le mode Compose démarre l'application, Prometheus, Grafana et Evidently. Les ports sont liés uniquement à l'interface locale Windows :

```powershell
docker compose up --build -d
docker compose ps
```

- Application : <http://127.0.0.1:8765>
- Grafana (identifiants initiaux `admin` / `admin`, à changer à la première connexion) : <http://127.0.0.1:3000>
- Prometheus : <http://127.0.0.1:9090>
- Evidently : <http://127.0.0.1:8000>

Le dashboard **Marso Robotics MLOps**, sa source Prometheus et trois règles d'alerte (application indisponible, aucun modèle promu, pipeline en échec récent) sont provisionnés automatiquement. Prometheus affiche l'état des alertes, mais aucun Alertmanager ni canal de notification n'est inclus dans ce prototype. Arrêt sans supprimer les volumes :

```powershell
docker compose down
```

Les données de l'application sont dans `data/`, les logs dans `logs/`, les volumes Grafana et Prometheus sont nommés et persistants. Ne pas utiliser `docker compose down -v` sauf si la suppression des données de monitoring est voulue. Le démarrage Compose construit une image Python avec OpenCV, scikit-learn et Evidently, puis télécharge Grafana et Prometheus : surveiller l'espace disque Docker Desktop et supprimer manuellement les ressources inutilisées après confirmation. La limite Prometheus porte sur sa base TSDB, pas sur l'espace réservé par le disque virtuel Docker.

## Orchestration et observabilité locales

La section « Orchestration et supervision » permet de lancer le pipeline synthétique à la demande ou de le planifier toutes les 1, 5, 15 ou 60 minutes. L'intervalle accepté par l'API est de 60 secondes à 24 heures. Désactiver le planning n'arrête pas une exécution déjà démarrée ; un lancement concurrent du même job est refusé. La configuration survit au redémarrage. Au redémarrage après une interruption pendant un job, l'exécution restée `running` est marquée en échec et journalisée ; aucune reprise automatique des données n'est prétendue.

- `GET /api/health` : disponibilité et état du scheduler.
- `GET /api/operations` : fréquence, prochaine exécution, historique et logs récents. Les requêtes GET de rafraîchissement de cette route restent mesurées dans Prometheus, mais ne sont pas ajoutées aux logs JSON pour éviter que la consultation des journaux ne les modifie elle-même.
- `GET /metrics` : métriques Prometheus HTTP et pipeline, avec labels bornés (route, méthode, statut, job, état).
- `GET /api/ml/status` : modèle actif, dernières versions et lots prédits.
- `POST /api/ml/train` : entraîne/évalue l'Isolation Forest sur le CSV synthétique ; seuils de promotion AUC ≥ 0,75 et F1 ≥ 0,5.
- `POST /api/ml/predict` : prédit sur un CSV avec `duration_ms`, `battery_pct`, `x_m`, `y_m` et crée un rapport Evidently de dérive.
- Interface Evidently : snapshots de dérive stockés dans `data/evidently/`, avec `include_data=False` pour ne pas archiver les lignes importées.
- `logs/app.jsonl` : événements JSON horodatés incluant ID de requête, durée, statut HTTP ou ID/durée d'exécution. Aucun contenu des fichiers importés ni données de télémétrie n'est inscrit dans les logs.
- `data/operations.sqlite3` : configuration et historique synthétique des runs.
- `data/mlops.sqlite3` et `data/models/` : registre local, métriques d'évaluation et 3 artefacts de modèles promus maximum.

L'entraînement exige une colonne `is_anomaly` contenant les classes 0 et 1. Le score est mesuré sur un découpage test stratifié ; un candidat qui n'atteint pas les seuils reste rejeté et ne remplace pas le modèle en service. Les labels du jeu fourni sont artificiels et servent uniquement à tester le câblage, pas à démontrer la qualité sur des robots réels. L'Isolation Forest et le seuil de contamination doivent être recalibrés avec un jeu de référence annoté représentatif avant usage opérationnel. Evidently détecte des changements de distribution, pas une dérive causale ni une dégradation de qualité certaine.

L'ordonnanceur maison est délibérément limité à **un processus sur un seul PC** : il n'a ni retry configurable, ni exécution distribuée, ni gestion de dépendances de tâches, ni alerte externe. Il convient pour essayer les concepts sur Windows, pas pour coordonner des pipelines de production.

## Formats et traitement

- CSV avec colonnes recommandées `timestamp`, `robot_id`, `event`, `duration_ms`, `battery_pct`, `x_m`, `y_m`.
- JSONL avec un objet JSON par ligne. Les colonnes optionnelles manquantes sont ignorées ; maximum 25 Mo et 200 000 lignes par analyse.
- Vidéo MP4, AVI, MOV ou MKV, maximum 250 Mo ; le résultat est un MP4 à télécharger.
- Pixelisation des visages détectés par YuNet (OpenCV Zoo), sur chaque image et quatre recadrages agrandis, avec un seuil de confiance de 0,35 et suppression des détections en double. Sur les images sombres (luminance moyenne < 90), une passe supplémentaire est faite sur une copie rehaussée par CLAHE. Une marge (20 % sur les côtés, 25 % en haut et en bas) protège les contours autour de chaque visage. Un suivi par recouvrement des boîtes extrapole brièvement le mouvement et conserve le masque jusqu'à 0,25 s (au moins 2 images) après une détection pour combler des images manquées. Les zones éloignées des visages restent nettes. La mosaïque est plafonnée à 8 blocs par côté afin que les grands visages restent méconnaissables. Cette recherche multi-échelle augmente le temps de traitement. Le modèle ONNX (~233 Ko) est téléchargé au premier traitement visage, vérifié par SHA-256 et réutilisé depuis `data/models/`.
- Le profil qualité compte les horodatages, identifiants robot et types d'événement manquants, ainsi que les batteries hors de l'intervalle 0–100 et les durées négatives. Ce sont des alertes descriptives ; aucun seuil métier ni correction automatique n'est inféré.

**Important — confidentialité :** YuNet est plus robuste que l'ancien classificateur Haar frontal, mais peut encore manquer les visages masqués, minuscules, flous, de profil extrême ou hors champ. La marge et le maintien temporel couvrent les contours et de courtes pertes de détection, mais ne retrouvent pas un visage jamais détecté et peuvent laisser un masque brièvement à l'emplacement prédit après la sortie du visage. Les zones éloignées des visages restent nettes. L'anonymisation automatique seule ne garantit pas la conformité RGPD : inspecter toute la vidéo image par image et conserver l'original dans un espace restreint. Le prototype ne collecte pas de biométrie et n'envoie pas les vidéos au cloud ; seul le poids YuNet peut être téléchargé du dépôt OpenCV Zoo.

### Choix du détecteur et références scientifiques

La requête Google Scholar fournie met notamment en avant SecurePose (Bajpai & Aravamuthan, *Scientific Reports*, 2026). Sur 116 vidéos cliniques de marche d'enfants atteints de paralysie cérébrale, l'article rapporte que son système dépasse six méthodes de détection comparées, avec une précision de désidentification comparable à un floutage manuel et un temps inférieur de 91,08 %. Ce résultat concerne des vidéos cliniques et ne garantit pas les mêmes performances sur des caméras d'entrepôt. Il confirme surtout l'importance d'évaluer la détection sur le domaine et les séquences réels, plutôt que de compter sur un filtre de flou seul.

Pour rester léger, hors ligne après le premier téléchargement et simple à installer sur Windows, le prototype utilise YuNet via l'API DNN intégrée à OpenCV au lieu d'ajouter un framework de modèles et plusieurs centaines de mégaoctets de poids. OpenCV Zoo publie un modèle YuNet MIT de 232 589 octets ; sa fiche rapporte AP 0,8844 (facile), 0,8656 (moyen) et 0,7503 (difficile) sur WIDER Face. Le seuil de confiance a été abaissé à 0,35 et quatre recadrages agrandis par image visent à mieux couvrir les petits visages ; le coût est jusqu'à six inférences par image (avec la passe basse lumière) et davantage de faux positifs possibles. Ces choix ne sont pas une mesure de rappel sur la vidéo fournie : il faut les comparer sur des images annotées du domaine réel. L'extension des rectangles et la mosaïque visent à réduire les contours non protégés autour des détections ; elles ne peuvent pas compenser une détection manquée.

- Bajpai, R. & Aravamuthan, B. (2026). [SecurePose: Automated face blurring and human movement kinematics extraction from videos recorded in clinical settings](https://doi.org/10.1038/s41598-026-50340-5), *Scientific Reports*, 16. [Notice et résumé indexés par Crossref](https://api.crossref.org/works/10.1038/s41598-026-50340-5).
- Wu, W., Peng, H. & Yu, S. (2023). [YuNet: A Tiny Millisecond-level Face Detector](https://doi.org/10.1007/s11633-023-1423-y), *Machine Intelligence Research*, 20, 656–665.
- [OpenCV Zoo — YuNet, modèle, résultats et licence MIT](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

## Tests

```powershell
python -m pip install pytest
python -m pytest -q
```

## Proposition d'architecture cible pour plusieurs To/jour

Ce prototype n'est **pas** le pipeline de production : un serveur Python local et des fichiers ZIP ne peuvent pas absorber durablement plusieurs To/jour. Conserver le même contrat de données, puis faire évoluer les composants séparément :

| Besoin | Option open source | Responsabilité |
| --- | --- | --- |
| Ingestion robot → site | MQTT (EMQX ou Mosquitto) + Redpanda/Kafka | Buffer, partitionnement par site/robot, rejeu et contrôle de débit |
| Données brutes immuables | MinIO/S3 on-prem + Apache Iceberg | Zone bronze, rétention, chiffrement, catalogue et évolution de schéma |
| Traitements batch/stream | Apache Flink (streaming) et Spark (batch) | Validation, déduplication, enrichissement et données argent/or |
| Orchestration | Dagster | Pipelines idempotents, reprises, suivi des jeux et versions de traitement |
| SQL et séries temporelles | Trino + PostgreSQL/TimescaleDB | Analyse exploratoire, métadonnées, états récents et KPI |
| Plateforme équipe | FastAPI + React ou Vue, RBAC OIDC | API opérationnelle, recherches, annotations et exports |
| Observabilité | OpenTelemetry + Prometheus + Grafana | Latence, retard de consommation, qualité, coûts et alertes |
| Anonymisation vidéo | FFmpeg + détecteur de visages validé/déployé localement, file de quarantaine | Zones d'accès restreint, revue humaine, traçabilité et règles de rétention |

À plusieurs To/jour, le débit, la rétention, le nombre de sites, la connectivité, la politique de conservation vidéo et les exigences RGPD déterminent le dimensionnement. Ne pas promettre une capacité chiffrée avant un test de charge avec des profils représentatifs ; garder les objets vidéo dans le stockage objet et leur index/métadonnées dans le SQL.

### Pratiques retenues et sources consultées (23 octobre 2026)

- **Instrumentation en trois signaux complémentaires :** logs pour les événements explicables, métriques pour les tendances/alertes et traces pour relier les étapes d'une requête ou d'un pipeline distribué. Le prototype a les logs JSON, les durées et compteurs Prometheus ; les traces distribuées ne sont pas justifiées dans ce processus local à un service.
  - [OpenTelemetry — Signals](https://opentelemetry.io/docs/concepts/signals/)
  - [OpenTelemetry — Logs Data Model](https://opentelemetry.io/docs/specs/otel/logs/data-model/)
- **Mesures adaptées aux jobs batch :** état de la dernière réussite, runtime, dernière fin même en échec, volumes traités et durée des étapes. Prometheus recommande en plus un heartbeat de bout en bout pour déceler un flux bloqué ; ce prototype n'a pas de flux entrant, donc le heartbeat est à ajouter avec l'ingestion réelle.
  - [Prometheus — Instrumentation practices](https://prometheus.io/docs/practices/instrumentation/)
  - [Prometheus — Metric naming](https://prometheus.io/docs/practices/naming/)
- **Planification et fraîcheur des données :** Dagster documente l'automatisation de jobs et les politiques d'asset freshness, tandis qu'Airflow permet de déclencher des DAGs sur la mise à jour d'assets plutôt qu'uniquement sur une horloge. Pour la production, préférer des déclenchements sur disponibilité des données et des seuils de retard, en plus du cron.
  - [Dagster — Schedules](https://docs.dagster.io/guides/automate/schedules)
  - [Dagster — Asset freshness policies](https://docs.dagster.io/guides/observe/asset-freshness-policies)
  - [Apache Airflow — Asset-aware scheduling](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/asset-scheduling.html)
- **Qualité mesurable et tests préventifs :** dbt décrit l'association de contrôles proactifs des transformations avec l'observabilité des incidents de données.
  - [dbt Labs (25 août 2025) — Building resilience: Observability for modern data teams](https://www.getdbt.com/blog/data-observability)
- **Travaux scientifiques très récents :** « Implementation Guidelines for Data Quality Metrics » (Jung et al., 7 octobre 2026) propose de rendre exécutables des métriques ISO de qualité et de distinguer les contrôles généralisables, paramétrés par des références propres, et ceux nécessitant un jugement humain. Il s'agit d'un **préprint arXiv soumis à EDBT 2027, pas d'un article évalué par les pairs publié** à la date de cette revue ; il motive ici de rendre les règles de qualité explicites et de ne pas apprendre de seuils sur le jeu synthétique.
  - [arXiv:2610.10919](https://arxiv.org/abs/2610.10919)
- **Surveillance en ligne des séries temporelles :** « TSGuard » (arXiv, octobre 2026) décrit une boucle de détection des valeurs absentes/retardées, validation par contraintes physiques, puis décision explicite de conserver ou corriger. Le preprint indique une présentation CIKM '26 à venir ; ce n'est pas une publication déjà parue au 23 octobre 2026.
  - [arXiv:2610.03147](https://arxiv.org/abs/2610.03147)

Ces sources étayent des motifs d'ingénierie, pas un dimensionnement de capacité : les préprints ne valent pas validation industrielle et l'article dbt est une source fournisseur, non une étude indépendante. Pour les données robot réelles, définir avec les équipes des contrôles de plausibilité physique (batterie, vitesses, cadence, horodatage, synchronisation) et faire valider leurs seuils avant de déclencher automatiquement des corrections.
