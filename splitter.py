from splitters.clause_splitter import ClauseSplitter


if __name__ == '__main__':

    sentences = [
        # conj verbale soggetto implicito
        "He came home, took a shower and immediately went to bed.",
        # conj verbale soggetto esplicito
        "He met her at the station and he kissed her.",
        # nominal conj con modificatori condivisi
        "American and British professors and students are very good.",
        # nominal conj + relcl
        "American and British professors and students who love something are very good.",

        # relcl + advcl
        "The book that John wrote became famous because it inspired many readers.",
        "The woman who looked happy danced when the music started.",
        # relcl + acl
        "The scientist who discovered the cure had a chance to save millions.",
        "The painting which the museum bought had a story to tell.",
        # advcl + acl
        "She had a decision to make because her boss resigned.",
        "He found a way to escape before the door closed.",
        # relcl + relcl
        "The man I met introduced me to the woman who won the prize.",
        "The book that she wrote inspired the student who solved the problem.",
        # tutti e tre insieme
        "The scientist who discovered the cure had a chance to publish because the journal accepted his work.",
        "The movie that we watched had a scene to remember because it moved everyone.",
    ]

    splitter = ClauseSplitter()

    for s in sentences:
        splits = splitter.split_sentence(s)
        print("\nSentence:", s)
        for split in splits:
            print(" -", split)