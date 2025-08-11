def generate_script_from_title(title):
    # Script di esempio simulato (finto Claude)
    return f"""Titolo: {title}

Ciao e benvenuto sul canale! Oggi parliamo di: "{title}".

In questo video, ti guiderò passo passo attraverso tutto quello che c’è da sapere su questo argomento.
Scoprirai curiosità, strategie, e suggerimenti pratici per migliorare la tua vita quotidiana.

Pronto a cominciare? Allora partiamo subito!

[Segmento 1: Introduzione]
...

[Segmento 2: Approfondimento]
...

[Segmento 3: Conclusione]
Grazie per aver seguito il video! Se ti è piaciuto, lascia un like e iscriviti per non perderti i prossimi contenuti."""

# Quando eseguito da terminale
if __name__ == "__main__":
    titolo = input("🎬 Inserisci il titolo del video: ")
    script = generate_script_from_title(titolo)

    # Salviamo in un file txt
    filename = titolo.replace(" ", "_").lower() + ".txt"
    output_path = f"data/outputs/scripts/{filename}"

    with open(output_path, "w") as f:
        f.write(script)

    print("\n✅ Script generato e salvato in:")
    print(output_path)

