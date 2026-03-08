from g2p_en import G2p

# import nltk
# nltk.download('averaged_perceptron_tagger_eng') # just for the first time to download this tagger

texts = ["bookkepaer",
         "The pipes are made of lead.",
         "She will lead the team to victory",
         "Please tear the paper in half",
         "A tear rolled down her cheek."
] 
g2p = G2p()
for text in texts:
    out = g2p(text)
    print(out)