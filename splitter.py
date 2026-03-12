from splitters.clause_splitter import ClauseSplitter


if __name__ == '__main__':

    sentences = [

        # advcl con advmod annidato
        "She left early because she felt sick.",
        "He ran faster because he trained harder.",

        # advcl con relcl annidato
        "He succeeded because the mentor who guided him was excellent.",
        "She failed because the machine that she used was broken.",

        # advcl con acl annidato
        "He won because he had a strategy to follow.",
        "She cried because she had no reason to stay.",

        # ccomp con advcl annidato
        "She thinks that he left because he was angry.",
        "He believes that she stayed because she loved him.",

        # ccomp con relcl annidato
        "He said that the woman who called was his sister.",
        "She knows that the car that he bought was expensive.",

        # ccomp con acl annidato
        "He said that she had a plan to escape.",
        "She believes that they found a cure to test.",

        # relcl con advcl annidato
        "The man who left because he was angry never returned.",
        "The student who failed because she missed the exam asked for help.",

        # acl con advcl annidato
        "She had a plan to escape before the guards arrived.",
        "He found a way to win before the time ran out.",

        # conj con ccomp
        "He said that she left and that the door was open.",
        "She believes that he is innocent and that the judge agreed.",

        # nominal conj con advcl
        "The professors and students protested because the exam was unfair.",
        "The doctors and nurses worked hard because the situation was critical.",

        # nominal conj con ccomp
        "The teachers and students believe that the system needs reform.",
        "The managers and employees agreed that the policy was wrong.",

        # nominal conj con relcl + ccomp
        "The professors and students who attended said that the lecture was inspiring.",
        "The doctors and nurses who worked said that the conditions were difficult.",

        # tutto insieme
        "The manager who hired him said that he had a chance to succeed because the team that she built was strong.",
        "The teacher who failed her believed that she had a reason to appeal because the exam that he wrote was unfair.",
    ]

    splitter = ClauseSplitter()

    for s in sentences:
        splits = splitter.split_sentence(s)
        print("\nSentence:", s)
        for split in splits:
            print(" -", split)