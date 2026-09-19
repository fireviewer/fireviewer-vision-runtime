# fireviewer-vision-runtime

## Repères documentaires — 19 septembre 2026

- **Rôle :** Inférence spécialisée : détection, pointing, segmentation et extraction de keyframes.
- **Statut :** Actif — package v0.1.1. Modèles externes/poids séparés de l’installation de base.
- **Entrées :** Images/vidéos référencées et checkpoints explicitement sélectionnés.
- **Sorties :** Observations en espace image : boxes, masques, points, keyframes et métadonnées.
- **Limites :** Une box ou un point image n’est pas une coordonnée terrain. Les tests CPU ne qualifient pas un modèle GPU ou sa performance terrain.

[Fiche du dépôt](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/repositories/fireviewer-vision-runtime.md) · [Architecture](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/ARCHITECTURE.md) · [Statuts et vocabulaire](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/STATUTS_ET_VOCABULAIRE.md).

Cette revue documentaire ne renouvelle aucun test ni aucune réception. Les procédures, versions et preuves techniques ci-dessous conservent leur périmètre et leur date.

> **Source active FV · private.** Détection, pointage, segmentation et extraction de keyframes. Voir [où travailler, quoi commiter et comment reprendre](ORGANISATION.md).

Specialized model inference and video keyframe extraction.

Python package: `fireviewer_vision_runtime`. Version: `0.1.1`.

## Installation

Install the versioned release wheels (including private FireViewer dependencies) from the release bundle. No sibling source checkout is required.

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

Les dépendances de base sont verrouillées avec hashes dans `requirements.lock.txt` (Python 3.13). Installer les wheels privés du même bundle via `--find-links`. Les extras lourds restent liés à leurs versions existantes et ne qualifient aucun GPU. Les tests de composant et leurs dépendances de test sont recensés dans le dossier unique de migration.
