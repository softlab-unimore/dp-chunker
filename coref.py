from fastcoref import FCoref

model = FCoref(device='cpu')

text = "I want to buy some apples. Which ones do you recommend?"

# chiedi a fastcoref di risolvere il testo
preds = model.predict([text])

# ottieni la versione risolta
resolved = preds[0].get_clusters(as_strings=True)

print("Clusters:", resolved)