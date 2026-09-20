# fireviewer-vision-runtime

## Repères documentaires — 19 septembre 2026

- **Rôle :** Inférence spécialisée : détection, pointing, segmentation et extraction de keyframes.
- **Statut :** Actif — package v0.1.1. Modèles externes/poids séparés de l’installation de base.
- **Entrées :** Images/vidéos référencées et checkpoints explicitement sélectionnés.
- **Sorties :** Observations en espace image : boxes, masques, points, keyframes et métadonnées.
- **Limites :** Une box ou un point image n’est pas une coordonnée terrain. Les tests CPU ne qualifient pas un modèle GPU ou sa performance terrain.

[Fiche du dépôt](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/repositories/fireviewer-vision-runtime.md) · [Architecture](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/ARCHITECTURE.md) · [Statuts et vocabulaire](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/STATUTS_ET_VOCABULAIRE.md).

Cette revue documentaire ne renouvelle aucun test ni aucune réception. Les procédures, versions et preuves techniques ci-dessous conservent leur périmètre et leur date.

> **Source active FV · public.** Détection, pointage, segmentation et extraction de keyframes. Voir [où travailler, quoi commiter et comment reprendre](ORGANISATION.md).

Specialized model inference and video keyframe extraction.

Python package: `fireviewer_vision_runtime`. Version: `0.1.1`.

## Installation

Install the versioned release wheels (including versioned FireViewer dependencies) from the release bundle. No sibling source checkout is required.

```sh
python -m pip install --find-links /path/to/release/wheels fireviewer-vision-runtime==0.1.1
python -m pytest tests -q
```

Optional model/provider environments are separate extras and retain their existing upstream constraints. Model weights, credentials, datasets and local evidence are external inputs.

## Ownership and compatibility

Target account: `fireviewer`. Target stewardship: Association FIRE-VIEWER. Historical authorship and AGPL-3.0-or-later notices are retained. This technical extraction is not a signed assignment of rights.

Source correspondence and hashes are recorded in the migration dossier. Existing schema IDs, algorithm revisions and evidence/publication gates are preserved. The former `firewarning_worker` or backend module paths are compatibility adapters in their original repository.

## Delivery boundary

Docker was deferred during the initial source delivery. The resumed private container phase, pinned images and acceptance limits are documented in [fireviewer-docker](https://github.com/fireviewer/fireviewer-docker). Production deployment remains separate. CPU/schema tests do not qualify GPU, visual or scientific performance.

## Sources et commandes propres au composant

Commande CPU de sélection vidéo : `fireviewer-keyframes`. Détection, pointage et segmentation restent des adaptateurs de modèles externes. Les extras de modèle sont distincts de l’installation de base ; aucun poids n’est téléchargé à l’import.

Les dépendances de base sont verrouillées avec hashes dans `requirements.lock.txt` (Python 3.13). Installer les wheels versionnés du même bundle via `--find-links`. Les extras lourds restent liés à leurs versions existantes et ne qualifient aucun GPU. Les tests de composant et leurs dépendances de test sont recensés dans le dossier unique de migration.

## Ouverture du code source — 19 septembre 2026

Ce dépôt fait partie du premier lot de huit composants FIRE-VIEWER ouvert au public sur décision du mainteneur. Le code original reste sous **AGPL-3.0-or-later** et la documentation originale sous **CC BY 4.0**, avec les notices et droits tiers existants.

Cette ouverture porte sur le code, son historique et les artefacts de développement déjà associés au dépôt. Les services déployés, comptes, données, corpus, modèles, secrets et autorisations des ressources externes gardent leur propre périmètre. Les sources des sites, du backend, des applications Android et de l’infrastructure restent privées. La visibilité publique ne constitue ni une nouvelle recette fonctionnelle ni un acte de cession des droits.

[Inventaire et périmètre d’ouverture](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/OPEN_SOURCE.md).

## Migration Bonsaï 2

Voir [le changement du juge et son état de validation](docs/BONSAI2.md).
