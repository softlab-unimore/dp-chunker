from splitters.clause_splitter import ClauseSplitter


if __name__ == '__main__':

    sentences = [
    #
    #     # advcl con advmod annidato
    #     "She left early because she felt sick.",
    #     "He ran faster because he trained harder.",
    #
    #     # advcl con relcl annidato
    #     "He succeeded because the mentor who guided him was excellent.",
    #     "She failed because the machine that she used was broken.",
    #
    #     # advcl con acl annidato
    #     "He won because he had a strategy to follow.",
    #     "She cried because she had no reason to stay.",
    #
    #     # ccomp con advcl annidato
    #     "She thinks that he left because he was angry.",
    #     "He believes that she stayed because she loved him.",
    #
    #     # ccomp con relcl annidato
    #     "He said that the woman who called was his sister.",
    #     "She knows that the car that he bought was expensive.",
    #
    #     # ccomp con acl annidato
    #     "He said that she had a plan to escape.",
    #     "She believes that they found a cure to test.",
    #
    #     # relcl con advcl annidato
    #     "The man who left because he was angry never returned.",
    #     "The student who failed because she missed the exam asked for help.",
    #
    #     # acl con advcl annidato
    #     "She had a plan to escape before the guards arrived.",
    #     "He found a way to win before the time ran out.",
    #
    #     # conj con ccomp
    #     "He said that she left and that the door was open.",
    #     "She believes that he is innocent and that the judge agreed.",
    #
    #     # nominal conj con advcl
    #     "The professors and students protested because the exam was unfair.",
    #     "The doctors and nurses worked hard because the situation was critical.",
    #
    #     # nominal conj con ccomp
    #     "The teachers and students believe that the system needs reform.",
    #     "The managers and employees agreed that the policy was wrong.",
    #
    #     # nominal conj con relcl + ccomp
    #     "The professors and students who attended said that the lecture was inspiring.",
    #     "The doctors and nurses who worked said that the conditions were difficult.",
    #
    #     # tutto insieme
    #     "The manager who hired him said that he had a chance to succeed because the team that she built was strong.",
    #     "The teacher who failed her believed that she had a reason to appeal because the exam that he wrote was unfair.",
    #
        # relcl + advcl + ccomp
        "The scientist who discovered the cure said that he succeeded because he worked hard.",
        "The teacher who failed her believed that she cheated because the answers were identical.",

        # relcl + acl + advcl
        "The man who had a plan to escape left before the guards arrived.",
        "The woman who found a way to survive waited until the storm passed.",

        # ccomp + relcl + advcl
        "She told me that the doctor who treated him left because the hospital closed.",
        "He said that the lawyer who defended her won because the evidence was clear.",

        # nominal conj + relcl + advcl
        "The doctors and nurses who treated him worked harder because the situation was critical.",
        "The professors and students who attended protested because the rules were unfair.",

        # nominal conj + ccomp + advcl
        "The managers and employees said that the policy was wrong because it hurt everyone.",
        "The teachers and parents believed that the system failed because nobody acted.",

        # ccomp + ccomp + relcl
        "She knows that he believes that the woman who called was lying.",
        "He said that she thinks that the man who left never returned.",

        # relcl + relcl + ccomp
        "The book that the author who won the prize wrote inspired millions.",
        "The cure that the scientist who worked alone discovered saved thousands.",

        # advcl + acl + ccomp
        "He succeeded because he had a strategy to follow that his mentor recommended.",
        "She failed because she missed a chance to prepare that her teacher offered.",

        # conj + ccomp + relcl
        "He came home and said that the woman who called was his sister.",
        "She left early and told me that the man who followed her was dangerous.",

        # tutto insieme
        "The scientist who discovered the cure said that he had a chance to publish because the journal that she recommended accepted his work.",
        "The teacher who inspired her believed that she had a reason to succeed because the school that he founded supported its students.",
        "The managers and employees who attended said that they found a way to solve the problem because the consultant that the board hired was excellent.",
    ]

    splitter = ClauseSplitter()

    for s in sentences:
        splits = splitter.split_sentence(s)
        print("\nSentence:", s)
        for split in splits:
            print(" -", split)