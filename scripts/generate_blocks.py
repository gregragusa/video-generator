import os

def split_script_into_blocks(script_text, max_block_length=300):
    # Divide il testo in blocchi basati su lunghezza massima o segmenti
    blocks = []
    current_block = ""

    for line in script_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(current_block) + len(line) < max_block_length:
            current_block += line + " "
        else:
            blocks.append(current_block.strip())
            current_block = line + " "
    if current_block:
        blocks.append(current_block.strip())
    return blocks

# TEST manuale
if __name__ == "__main__":
    title = input("🎬 Titolo del file script: ")
    filename = title.replace(" ", "_").lower() + ".txt"
    script_path = f"data/outputs/scripts/{filename}"

    if not os.path.exists(script_path):
        print(f"❌ File non trovato: {script_path}")
        exit(1)

    with open(script_path, "r") as f:
        script_text = f.read()

    blocks = split_script_into_blocks(script_text)
    print(f"\n✅ {len(blocks)} blocchi trovati:\n")

    for i, block in enumerate(blocks, 1):
        print(f"[BLOCCO {i}]\n{block}\n")

    # Salva i blocchi in file separati
    output_dir = f"data/outputs/blocks/{title.replace(' ', '_').lower()}/"
    os.makedirs(output_dir, exist_ok=True)

    for i, block in enumerate(blocks, 1):
        with open(os.path.join(output_dir, f"block_{i}.txt"), "w") as f:
            f.write(block)

    print(f"\n📁 Blocchi salvati in: {output_dir}")

