from datasets import load_dataset
from pathlib import Path

CLASSES_VOULUES = {
    1: "releve_bancaire",    # budget
    4: "bon_de_commande",    # form
    7: "contrat",            # letter
}

OUTPUT_DIR = "data/raw/rvlcdip"
LIMITE = 500

print("Téléchargement RVL-CDIP via chainyo/rvl-cdip...")

dataset = load_dataset(
    "chainyo/rvl-cdip",
    split="train",
    streaming=True
)

compteurs = {nom: 0 for nom in CLASSES_VOULUES.values()}

for exemple in dataset:
    label_id = exemple["label"]

    if label_id in CLASSES_VOULUES:
        nom_classe = CLASSES_VOULUES[label_id]

        if compteurs[nom_classe] >= LIMITE:
            continue

        dossier = Path(OUTPUT_DIR) / nom_classe
        dossier.mkdir(parents=True, exist_ok=True)

        img = exemple["image"]
        img_path = dossier / f"{nom_classe}_{compteurs[nom_classe]:04d}.png"
        img.save(str(img_path))
        compteurs[nom_classe] += 1

        if compteurs[nom_classe] % 50 == 0:
            print(f"  {nom_classe}: {compteurs[nom_classe]}/{LIMITE}")

    if all(v >= LIMITE for v in compteurs.values()):
        print("\nToutes les classes complètes !")
        break

print("\nRésultat final :")
for classe, count in compteurs.items():
    print(f"  {classe}: {count} images")